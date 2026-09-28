package posegraph

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Store is the concurrency-safe, versioned entry point of the package.
//
// Mutual exclusion makes each commit atomic: readers that take RLock observe a
// single complete version, either before or after a commit, and the Result
// returned by Query is a deep copy whose contents never mutate. Solver
// iterations happen on a private copy, so no intermediate state is ever
// visible externally.
type Store struct {
	mu        sync.RWMutex
	cond      *sync.Cond
	cfg       Config
	graph     *Graph
	pending   []Constraint
	rejected  []Constraint
	last      *Result
	version   int64
	seq       int64
	clock     func() int64
	persister *snapshotPersister
	closed    bool
}

// ErrStoreClosed is returned when mutating or optimizing a closed store.
var ErrStoreClosed = errors.New("posegraph: store is closed")

// NewStore creates a store. With cfg.SnapshotPath set, it restores the most
// recent consistent snapshot; every accepted commit is then persisted
// atomically so a restart loses no accepted constraints.
func NewStore(cfg Config) (*Store, error) {
	cfg.normalize()
	s := &Store{
		cfg:       cfg,
		graph:     newGraph(),
		clock:     clockNow,
		persister: newSnapshotPersister(cfg.SnapshotPath),
	}
	s.cond = sync.NewCond(&s.mu)
	if cfg.SnapshotPath != "" {
		snap, found, err := s.persister.load()
		if err != nil {
			return nil, fmt.Errorf("posegraph: load snapshot: %w", err)
		}
		if found {
			if err := s.restore(snap); err != nil {
				return nil, err
			}
		}
	}
	return s, nil
}

// Config returns the configuration in use.
func (s *Store) Config() Config {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg
}

// Submit applies one node or constraint atomically.
func (s *Store) Submit(ch Change) (CommitReport, error) {
	return s.Commit([]Change{ch})
}

// Commit applies a batch atomically. Processing order is deterministic and
// independent of the order of the input slice: node changes are sorted by ID,
// then constraint changes by a canonical (kind, endpoints, measurement) key.
// With AutoOptimize enabled the batch is optimized before the lock releases.
func (s *Store) Commit(changes []Change) (CommitReport, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return CommitReport{}, ErrStoreClosed
	}
	before := s.version
	report := CommitReport{VersionBefore: before}

	ordered := sortChanges(changes)
	for _, ch := range ordered {
		switch ch.Kind {
		case ChangeNode:
			report.Reports = append(report.Reports, s.applyNode(ch.Node))
		case ChangeConstraint:
			report.Reports = append(report.Reports, s.applyConstraint(ch.Constraint))
		default:
			report.Reports = append(report.Reports, ChangeReport{
				Change: ch, Outcome: OutcomeInvalid, Detail: "unknown change kind",
			})
		}
	}
	// New nodes may unlock previously dangling constraints; retry pending.
	if len(s.pending) > 0 {
		report.Reports = append(report.Reports, s.retryPending()...)
	}

	changed := false
	stateChanged := false
	for _, r := range report.Reports {
		switch r.Outcome {
		case OutcomeAdded, OutcomeFused, OutcomeLoopAccept, OutcomeLoopDownweight:
			changed = true
			stateChanged = true
		case OutcomePending, OutcomeRejected:
			// These mutate pending/rejected buffers even though they do not
			// produce a new estimate version.
			stateChanged = true
		}
	}
	if changed {
		diag, published := s.optimizeLocked()
		report.Optimization = &diag
		// The version advances only when a new consistent result is produced;
		// rejected loops and pure repeats that changed nothing keep version.
		if published {
			s.version++
		}
	}
	if changed || stateChanged {
		if err := s.persistLocked(); err != nil {
			return report, fmt.Errorf("posegraph: persist: %w", err)
		}
	}
	report.VersionAfter = s.version
	return report, nil
}

// Optimize forces an optimization run and publishes a new result on success.
func (s *Store) Optimize() (Diagnostics, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return Diagnostics{}, ErrStoreClosed
	}
	diag, published := s.optimizeLocked()
	if published {
		s.version++
	}
	if err := s.persistLocked(); err != nil {
		return diag, fmt.Errorf("posegraph: persist: %w", err)
	}
	s.cond.Broadcast()
	return diag, nil
}

// Query returns a deep copy of the latest successful result, or nil.
func (s *Store) Query() *Result {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneResult(s.last)
}

// WaitForVersion blocks until a successful result with version >= v exists or
// the store closes. The bool is false when the store closed first.
func (s *Store) WaitForVersion(v int64) (*Result, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for (s.last == nil || s.last.Version < v) && !s.closed {
		s.cond.Wait()
	}
	// If the requested version is now visible return it; otherwise the store
	// closed before it could ever be produced.
	if s.last != nil && s.last.Version >= v {
		return cloneResult(s.last), true
	}
	return cloneResult(s.last), false
}

// Version returns the current logical version.
func (s *Store) Version() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.version
}

// Snapshot returns an independent serializable copy of the full state.
func (s *Store) Snapshot() Snapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshotLocked()
}

// Close flushes persistence and rejects further mutations.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	s.cond.Broadcast()
	return s.persistLocked()
}

// ---------------------------------------------------------------------------
// internals (callers hold s.mu)
// ---------------------------------------------------------------------------

func (s *Store) applyNode(n Node) ChangeReport {
	if err := validateNode(n); err != nil {
		return ChangeReport{Change: Change{Kind: ChangeNode, Node: n},
			Outcome: OutcomeInvalid, Detail: err.Error()}
	}
	if existing, ok := s.graph.getNode(n.ID); ok {
		if nodesEqual(existing, n) {
			return ChangeReport{Change: Change{Kind: ChangeNode, Node: n},
				Outcome: OutcomeRepeated, Detail: "identical node already present"}
		}
		// Conflicting repeat: first submission wins deterministically. The
		// conflict is surfaced rather than silently altering the pose.
		return ChangeReport{Change: Change{Kind: ChangeNode, Node: n},
			Outcome: OutcomeRepeated,
			Detail:  "conflicting node repeat ignored; first submission retained"}
	}
	n.FirstAdded = s.tick()
	s.graph.nodes[n.ID] = n
	return ChangeReport{Change: Change{Kind: ChangeNode, Node: n}, Outcome: OutcomeAdded}
}

func (s *Store) applyConstraint(c Constraint) ChangeReport {
	ch := Change{Kind: ChangeConstraint, Constraint: c}
	if c.ID == "" {
		c.ID = defaultConstraintID(c)
		ch.Constraint = c
	}
	if err := validateConstraint(c, func(id NodeID) bool {
		_, ok := s.graph.getNode(id)
		return ok
	}); err != nil {
		if errors.Is(err, errUnknownEndpoint) {
			s.pending = append(s.pending, c)
			return ChangeReport{Change: ch, Outcome: OutcomePending,
				Detail: "endpoint missing; constraint buffered"}
		}
		return ChangeReport{Change: ch, Outcome: OutcomeInvalid, Detail: err.Error()}
	}

	if existing, ok := s.graph.getConstraint(c.ID); ok {
		if constraintsCompatible(existing, c) {
			return ChangeReport{Change: ch, Outcome: OutcomeRepeated,
				Detail: "identical constraint already present"}
		}
		fused := fuseConflicts(existing, c, s.cfg.ConflictInflation)
		fused.FirstAdded = existing.FirstAdded
		s.graph.constraints[c.ID] = fused
		return ChangeReport{Change: ch, Outcome: OutcomeFused,
			Detail: "conflicting submission merged by symmetric information weighting"}
	}

	// Loop candidates are gated against the current best estimate and never
	// allowed to participate in the very evaluation that judges them.
	if c.Loop {
		current := s.currentPoses()
		decision := gateLoop(s.cfg, s.graph, current, c)
		if !decision.Accepted {
			c.Accepted = false
			c.Rejected = decision.Reason
			s.rejected = append(s.rejected, c)
			return ChangeReport{Change: ch, Outcome: OutcomeRejected,
				Detail: decision.Reason, Chi2PerDOF: decision.Chi2PerDOF}
		}
		c.Accepted = true
		c.RobustWeight = decision.Weight
		c.FirstAdded = s.tick()
		s.graph.constraints[c.ID] = c
		out := OutcomeLoopAccept
		if decision.Downweighted {
			out = OutcomeLoopDownweight
		}
		return ChangeReport{Change: ch, Outcome: out, Detail: decision.Reason,
			Weight: decision.Weight, Chi2PerDOF: decision.Chi2PerDOF}
	}

	c.Accepted = true
	if c.RobustWeight == 0 {
		c.RobustWeight = 1
	}
	c.FirstAdded = s.tick()
	s.graph.constraints[c.ID] = c
	return ChangeReport{Change: ch, Outcome: OutcomeAdded}
}

// retryPending re-validates buffered constraints after node additions. It
// preserves pending-buffer order but only removes entries that now resolve.
func (s *Store) retryPending() []ChangeReport {
	var reports []ChangeReport
	remaining := s.pending[:0]
	for _, c := range s.pending {
		ok := true
		if c.Kind == Relative {
			_, ok1 := s.graph.getNode(c.A)
			_, ok2 := s.graph.getNode(c.B)
			ok = ok1 && ok2
		} else {
			_, ok = s.graph.getNode(c.B)
		}
		if !ok {
			remaining = append(remaining, c)
			continue
		}
		reports = append(reports, s.applyConstraint(c))
	}
	s.pending = remaining
	return reports
}

// currentPoses returns the pose estimate used to gate new loop candidates:
// the latest successful result if available, otherwise node Initial poses.
func (s *Store) currentPoses() map[NodeID]Pose2 {
	if s.last != nil {
		poses := make(map[NodeID]Pose2, len(s.last.Nodes))
		for id, m := range s.last.Nodes {
			poses[id] = m.Pose
		}
		return poses
	}
	poses := make(map[NodeID]Pose2, s.graph.nodeCount())
	for _, id := range s.graph.sortedNodeIDs() {
		n, _ := s.graph.getNode(id)
		poses[id] = n.Initial
	}
	return poses
}

// optimizeLocked runs the solver on a private copy and publishes a Result only
// on an acceptable run. A failed/ill-conditioned/stagnated run never replaces
// a prior good result; diagnostics are returned either way, along with a flag
// telling whether a new result version was published.
func (s *Store) optimizeLocked() (Diagnostics, bool) {
	// Seed from the previous result for fast, stable incremental convergence.
	var warm map[NodeID]Pose2
	if s.last != nil {
		warm = make(map[NodeID]Pose2, len(s.last.Nodes))
		for id, m := range s.last.Nodes {
			warm[id] = m.Pose
		}
	}
	out := runOptimization(s.cfg, s.graph, warm)
	diag := out.diag
	if out.ok {
		res := &Result{
			Version:     s.version + 1,
			Nodes:       out.marginals,
			Residuals:   out.residuals,
			PerNode:     out.perNode,
			Diag:        out.diag,
			EstimatedAt: time.Now(),
		}
		s.last = res
		// Persist accepted loop robust weights back onto edges.
		for cid, wt := range out.loopWeights {
			if c, ok := s.graph.constraints[cid]; ok {
				c.RobustWeight = wt
				s.graph.constraints[cid] = c
			}
		}
		s.cond.Broadcast()
		return diag, true
	}
	diag.Warnings = append(diag.Warnings,
		"result not published: previous consistent result retained")
	return diag, false
}

func (s *Store) tick() int64 {
	s.seq++
	return s.seq
}

func (s *Store) snapshotLocked() Snapshot {
	nodes := make(map[NodeID]Node, len(s.graph.nodes))
	for id, n := range s.graph.nodes {
		nodes[id] = n
	}
	constraints := make(map[string]Constraint, len(s.graph.constraints))
	for id, c := range s.graph.constraints {
		constraints[id] = c
	}
	pending := append([]Constraint(nil), s.pending...)
	rejected := append([]Constraint(nil), s.rejected...)
	return Snapshot{
		Version:     s.version,
		Nodes:       nodes,
		Constraints: constraints,
		Pending:     pending,
		Rejected:    rejected,
		NextSeq:     s.seq,
		LastResult:  cloneResult(s.last),
		Config:      s.cfg,
		SavedAt:     time.Now(),
	}
}

func (s *Store) persistLocked() error {
	if s.cfg.SnapshotPath == "" {
		return nil
	}
	return s.persister.save(s.snapshotLocked())
}

func (s *Store) restore(snap Snapshot) error {
	s.cfg = snap.Config
	s.cfg.normalize()
	s.graph = newGraph()
	for id, n := range snap.Nodes {
		s.graph.nodes[id] = n
	}
	for id, c := range snap.Constraints {
		s.graph.constraints[id] = c
	}
	s.pending = append([]Constraint(nil), snap.Pending...)
	s.rejected = append([]Constraint(nil), snap.Rejected...)
	s.seq = snap.NextSeq
	s.version = snap.Version
	s.last = cloneResult(snap.LastResult)
	return nil
}

// cloneResult deep-copies a Result so callers can never mutate published state.
func cloneResult(r *Result) *Result {
	if r == nil {
		return nil
	}
	cp := *r
	cp.Nodes = make(map[NodeID]Marginal, len(r.Nodes))
	for k, v := range r.Nodes {
		cp.Nodes[k] = v
	}
	cp.Residuals = make(map[string]Residual, len(r.Residuals))
	for k, v := range r.Residuals {
		cp.Residuals[k] = v
	}
	cp.PerNode = make(map[NodeID]NodeResidual, len(r.PerNode))
	for k, v := range r.PerNode {
		cp.PerNode[k] = v
	}
	cp.Diag.Warnings = append([]string(nil), r.Diag.Warnings...)
	return &cp
}

// sortChanges canonicalizes batch order: nodes first (by ID, then initial
// pose), constraints after (by canonical key). This is why identical input
// sets produce identical results regardless of caller slice ordering.
func sortChanges(in []Change) []Change {
	out := append([]Change(nil), in...)
	keyConstraint := func(c Constraint) string {
		if c.Kind == Prior {
			return fmt.Sprintf("p:%s", c.B)
		}
		if c.A < c.B {
			return fmt.Sprintf("r:%s:%s", c.A, c.B)
		}
		return fmt.Sprintf("r:%s:%s", c.B, c.A)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		if out[i].Kind == ChangeNode {
			return out[i].Node.ID < out[j].Node.ID
		}
		ki := out[i].Constraint.ID
		kj := out[j].Constraint.ID
		if ki == "" {
			ki = defaultConstraintID(out[i].Constraint)
		}
		if kj == "" {
			kj = defaultConstraintID(out[j].Constraint)
		}
		if ki != kj {
			return ki < kj
		}
		return keyConstraint(out[i].Constraint) < keyConstraint(out[j].Constraint)
	})
	return out
}

// clockNow is the default wall clock hook; tests inject deterministic clocks.
func clockNow() int64 { return time.Now().UnixMicro() }
