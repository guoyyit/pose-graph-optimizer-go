package posegraph

import (
	"context"
	"sort"
	"sync"
)

// Service is the concurrency-safe, incremental, persistent pose graph API.
// All methods are safe for concurrent use. Readers always observe a complete
// published version, never a partially updated or mid-iteration state:
// every Query returns a pointer to an immutable *Result built atomically.
type Service struct {
	cfg   Config
	mu    sync.RWMutex
	graph *Graph

	version   int64
	current   *Result
	history   map[int64]*Result
	histOrder []int64
	histCap   int

	store store

	// gateRejected records loop edges rejected by the optimizer gate so that
	// later re-submissions stay consistent across incremental versions.
	gateRejected map[string]GateEvent

	mutationsSinceSnapshot int
}

// NewService builds a service. If cfg.PersistenceDir is set, state is
// recovered from the latest consistent snapshot plus WAL before returning.
func NewService(cfg Config) (*Service, error) {
	s := &Service{
		cfg:          cfg,
		graph:        NewGraph(),
		history:      map[int64]*Result{},
		histCap:      16,
		gateRejected: map[string]GateEvent{},
	}
	s.graph.dupChi2Threshold = cfg.DuplicateChi2Threshold
	if cfg.PersistenceDir != "" {
		st, err := newFileStore(cfg.PersistenceDir, cfg.SnapshotEvery)
		if err != nil {
			return nil, err
		}
		s.store = st
		if err := s.restore(); err != nil {
			return nil, err
		}
	}
	// Publish an initial version (even for an empty graph the result is
	// SolveEmpty, which is still a consistent readable version).
	s.republishLocked()
	return s, nil
}

// AddNode adds a node and republishes a new consistent version. It returns
// the optimizer error when the new version cannot be produced, but the
// previously published version stays queryable.
func (s *Service) AddNode(ctx context.Context, n Node) error {
	return s.mutate(ctx, walRecord{Kind: recNode, Node: &n}, func(g *Graph) error {
		g.AddNode(n)
		return nil
	})
}

// AddConstraint adds, merges or gates a constraint and republishes. A
// rejected loop closure returns ErrConstraintRejected but the decision is
// recorded; the rest of the graph remains usable. ErrPending means the
// constraint was buffered until both endpoints exist.
func (s *Service) AddConstraint(ctx context.Context, c Constraint) error {
	return s.mutate(ctx, walRecord{Kind: recConstraint, Constraint: &c}, func(g *Graph) error {
		_, err := g.AddConstraint(c)
		return err
	})
}

func (s *Service) mutate(ctx context.Context, rec walRecord, apply func(g *Graph) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	// Persist the intended mutation *before* publishing so a crash cannot
	// lose an accepted constraint.
	if s.store != nil {
		if err := s.store.appendRecord(rec); err != nil {
			return err
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var pendingErr error
	if perr := apply(s.graph); perr != nil {
		pendingErr = perr
	}
	// Transfer optimizer-level gate rejections back onto graph edge records
	// so they remain excluded in future versions (until re-gated).
	s.markGateStatesLocked()
	res := s.solveLocked()

	// Record gate decisions from diagnostics for future versions.
	for _, ev := range res.Diag.Rejected {
		s.gateRejected[ev.EdgeID] = ev
		if e, ok := s.graph.edges[ev.EdgeID]; ok {
			e.state = EdgeRejected
			e.weight = 0
			e.reason = ev.Reason
		}
	}
	for _, ev := range res.Diag.Downweighted {
		if e, ok := s.graph.edges[ev.EdgeID]; ok {
			e.state = EdgeDownweighted
		}
	}

	s.version++
	res.Version = s.version
	s.current = res
	s.keepVersion(res)

	// Snapshot bookkeeping.
	s.mutationsSinceSnapshot++
	snapErr := s.maybeSnapshotLocked()

	// Surface the most relevant error while keeping the version published.
	// Priority: a loop that THIS submission introduced and the gate rejected
	// is reported as a constraint rejection (the rest of the graph is fine);
	// otherwise a structurally/numerically failed solve; else pending I/O.
	if rec.Kind == recConstraint && rec.Constraint != nil {
		submittedID := rec.Constraint.ID
		if submittedID == "" {
			submittedID = autoID(*rec.Constraint)
		}
		if ev, rejected := s.gateRejected[submittedID]; rejected {
			_ = ev
			if e, ok := s.graph.edges[submittedID]; ok && e.state == EdgeRejected {
				return ErrConstraintRejected
			}
		}
	}
	if !res.Diag.OK && res.Diag.Status != SolveEmpty {
		return s.statusError(res.Diag)
	}
	if snapErr != nil {
		return snapErr
	}
	return pendingErr
}

func (s *Service) markGateStatesLocked() {
	for id, ev := range s.gateRejected {
		if e, ok := s.graph.edges[id]; ok {
			// A loop edge that was previously rejected stays rejected unless
			// its measurement changed (new edge ID would be generated).
			e.state = EdgeRejected
			e.weight = 0
			e.reason = ev.Reason
		}
	}
}

func (s *Service) solveLocked() *Result {
	// Work on a clone so an aborted solve never mutates published-adjacent
	// state; only accepted gate decisions are copied back afterwards.
	work := s.graph.clone()
	res := optimize(work, s.cfg, 0)
	// Copy effective downweight weights onto the live graph for continuity.
	for _, ev := range res.Diag.Downweighted {
		if e, ok := s.graph.edges[ev.EdgeID]; ok {
			for _, rr := range res.Residuals {
				if rr.EdgeID == ev.EdgeID {
					e.weight = rr.Weight
					break
				}
			}
		}
	}
	return res
}

func (s *Service) republishLocked() {
	res := s.solveLocked()
	s.version = max64(s.version, 1)
	res.Version = s.version
	s.current = res
	s.keepVersion(res)
}

func (s *Service) keepVersion(r *Result) {
	s.history[r.Version] = r
	s.histOrder = append(s.histOrder, r.Version)
	sort.Slice(s.histOrder, func(i, j int) bool { return s.histOrder[i] < s.histOrder[j] })
	for int64(len(s.histOrder)) > int64(s.histCap) {
		old := s.histOrder[0]
		s.histOrder = s.histOrder[1:]
		delete(s.history, old)
	}
}

// Query returns the latest consistent immutable result. The returned pointer
// is never mutated after publication.
func (s *Service) Query() *Result {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.current
}

// QueryAt returns a retained historical version, or nil if it has aged out.
func (s *Service) QueryAt(version int64) *Result {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.history[version]
}

// Versions returns the current monotonically increasing version number.
func (s *Service) Versions() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.version
}

// EdgeState reports the lifecycle state of a submitted constraint.
func (s *Service) EdgeState(id string) (EdgeState, string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if e, ok := s.graph.Edge(id); ok {
		return e.state, e.reason, true
	}
	return 0, "", false
}

// Close flushes a final snapshot and closes the persistence store.
func (s *Service) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.store == nil {
		return nil
	}
	if err := s.writeSnapshotLocked(); err != nil {
		s.store.Close()
		return err
	}
	return s.store.Close()
}

func (s *Service) statusError(d Diagnostics) error {
	switch d.Status {
	case SolveIllConditioned:
		return ErrDegenerate
	case SolveStagnated, SolveMaxIterations:
		return ErrNotConverged
	case SolveUnderconstrained:
		return ErrDegenerate
	case SolveHighResidual:
		return ErrNotConverged
	default:
		return nil
	}
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
