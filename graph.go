package posegraph

import (
	"fmt"
	"sort"
)

// ConstraintKind distinguishes odometry-style links from loop closures.
type ConstraintKind int

const (
	// KindOdometry is a sequential relative motion (trusted).
	KindOdometry ConstraintKind = iota
	// KindLoop is a loop-closure / revisiting constraint (gated).
	KindLoop
)

func (k ConstraintKind) String() string {
	switch k {
	case KindOdometry:
		return "odometry"
	case KindLoop:
		return "loop"
	default:
		return "unknown"
	}
}

// Node is a pose variable. Hint is an optional initial pose used when the
// node is not reachable from the anchor through accepted constraints.
type Node struct {
	ID   string
	Hint Pose
}

// Constraint is a relative pose measurement z = Log(T_i^{-1} * T_j) between
// nodes From and To. Provide exactly one of Covariance or Information;
// identity information is used when both are zero/omitted.
type Constraint struct {
	ID          string
	From        string
	To          string
	Measurement Pose
	Covariance  Mat3
	Information Mat3
	Kind        ConstraintKind
}

// EdgeState is the lifecycle state of a constraint inside the graph.
type EdgeState int

const (
	// EdgePending waits for one or both endpoints to be added.
	EdgePending EdgeState = iota
	// EdgeAccepted participates in optimization.
	EdgeAccepted
	// EdgeFused was merged into an identical duplicate constraint.
	EdgeFused
	// EdgeDownweighted participates with reduced weight (soft reject).
	EdgeDownweighted
	// EdgeRejected failed the consistency gate / conflict and is excluded.
	EdgeRejected
)

func (s EdgeState) String() string {
	switch s {
	case EdgePending:
		return "pending"
	case EdgeAccepted:
		return "accepted"
	case EdgeFused:
		return "fused"
	case EdgeDownweighted:
		return "downweighted"
	case EdgeRejected:
		return "rejected"
	default:
		return "unknown"
	}
}

// Edge is an internal, resolved constraint record.
type Edge struct {
	Constraint
	state     EdgeState
	weight    float64
	reason    string
	fusedInto string
}

func (e *Edge) State() EdgeState         { return e.state }
func (e *Edge) Weight() float64          { return e.weight }
func (e *Edge) Reason() string           { return e.reason }
func (e *Edge) writableWeight(w float64) { e.weight = w }

// Graph is the mutable, non-thread-safe container of nodes and edges.
type Graph struct {
	nodes map[string]*nodeData
	edges map[string]*Edge
	// pending holds constraints whose endpoints are not both present yet.
	pending map[string]*Edge
	// dupIndex maps an undirected endpoint pair to edge IDs on that pair.
	dupIndex map[string][]string
	nextSeq  int
	// dupChi2Threshold is the agreement gate for duplicate fusion.
	dupChi2Threshold float64
}

type nodeData struct {
	Node
}

// NewGraph returns an empty graph.
func NewGraph() *Graph {
	return &Graph{
		nodes:            map[string]*nodeData{},
		edges:            map[string]*Edge{},
		pending:          map[string]*Edge{},
		dupIndex:         map[string][]string{},
		dupChi2Threshold: DefaultConfig().DuplicateChi2Threshold,
	}
}

// pairKey builds an order-independent canonical key for an endpoint pair.
func pairKey(a, b string) string {
	if a > b {
		a, b = b, a
	}
	return a + "\x00" + b
}

// canonicalMeasurement returns the measurement expressed in the canonical
// (lexicographically smaller endpoint first) orientation, together with the
// canonical endpoint names.
func canonicalMeasurement(c Constraint) (lo, hi string, z Pose, info Mat3) {
	if c.From < c.To {
		return c.From, c.To, c.Measurement, c.Information
	}
	return c.To, c.From, Inverse(c.Measurement), c.Information
}

// autoID derives a deterministic ID from constraint content so identical
// resubmissions with empty IDs are naturally idempotent.
func autoID(c Constraint) string {
	lo, hi, z, info := canonicalMeasurement(c)
	return fmt.Sprintf("auto:%s|%s|%d|%.12g|%.12g|%.12g|%.12g|%.12g|%.12g|%.12g|%.12g|%.12g",
		lo, hi, c.Kind,
		z.X, z.Y, z.Theta,
		info[0], info[1], info[2], info[4], info[5], info[8])
}

// resolveInformation chooses the information matrix: explicit Information
// wins; otherwise invert a positive-definite Covariance; else identity.
func resolveInformation(c Constraint) (Mat3, error) {
	z := zero3()
	infoGiven := c.Information != z
	if infoGiven {
		// Accept but validate finiteness.
		for _, v := range c.Information {
			if v != v || v > 1e300 || v < -1e300 {
				return z, ErrInvalidConstraint
			}
		}
		return c.Information, nil
	}
	covGiven := c.Covariance != z
	if !covGiven {
		return identity3(), nil
	}
	inv, ok := symInverse(c.Covariance)
	if !ok {
		return z, ErrInvalidConstraint
	}
	return inv, nil
}

// measurementDisagreement returns the chi2 distance between two measurements
// on the same unordered pair, using the sum of their covariances as metric.
func measurementDisagreement(e1, e2 *Edge) float64 {
	var z1, z2 Pose
	var cov Mat3
	_, _, z1, _ = canonicalMeasurement(e1.Constraint)
	_, _, z2, _ = canonicalMeasurement(e2.Constraint)
	for _, e := range []*Edge{e1, e2} {
		c, ok := symInverse(e.Information)
		if !ok {
			c = identity3()
		}
		for k := 0; k < 9; k++ {
			cov[k] += c[k]
		}
	}
	d := Log(Compose(Inverse(z1), z2))
	cinv, ok := symInverse(cov)
	if !ok {
		return 0
	}
	return vecDot(d, matVec(cinv, d))
}

func infoStrength(e *Edge) float64 {
	return e.Information[0] + e.Information[4] + e.Information[8]
}

// AddNode inserts a node. Re-adding an existing node is idempotent: the
// stored hint is refreshed only if the existing one is the zero pose.
// Pending constraints that become complete are promoted and merged.
func (g *Graph) AddNode(n Node) { g.addNode(n) }

func (g *Graph) addNode(n Node) []string {
	promoted := []string{}
	if existing, ok := g.nodes[n.ID]; ok {
		z := Pose{}
		if existing.Hint == z && n.Hint != z {
			existing.Hint = n.Hint
		}
	} else {
		g.nodes[n.ID] = &nodeData{Node: n}
	}
	// Promote pending edges in deterministic (sorted ID) order.
	ids := make([]string, 0, len(g.pending))
	for id := range g.pending {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		e := g.pending[id]
		if g.HasNode(e.From) && g.HasNode(e.To) {
			delete(g.pending, id)
			g.installEdge(e)
			promoted = append(promoted, id)
		}
	}
	return promoted
}

// HasNode reports whether node id exists.
func (g *Graph) HasNode(id string) bool { _, ok := g.nodes[id]; return ok }

// AddConstraint registers a constraint. Pending constraints are buffered
// until both endpoints exist; duplicates are fused and conflicts resolved by
// deterministic rules independent of submission order. It returns the edge
// ID. ErrPending indicates buffering; nil means installed immediately.
func (g *Graph) AddConstraint(c Constraint) (string, error) {
	if c.From == "" || c.To == "" || c.From == c.To {
		return "", ErrInvalidConstraint
	}
	info, err := resolveInformation(c)
	if err != nil {
		return "", err
	}
	c.Information = info
	if c.ID == "" {
		c.ID = autoID(c)
	}
	e := &Edge{Constraint: c, state: EdgePending, weight: 1}

	// Idempotent resubmission of the same ID.
	if _, ok := g.edges[e.ID]; ok {
		return e.ID, nil
	}
	if _, ok := g.pending[e.ID]; ok {
		return e.ID, nil
	}

	if !g.HasNode(c.From) || !g.HasNode(c.To) {
		g.pending[e.ID] = e
		return e.ID, ErrPending
	}
	g.installEdge(e)
	return e.ID, nil
}

// installEdge applies the deterministic merge rules. Layered policy:
//
//   - Loop-closure edges never undergo graph-level conflict rejection. They
//     are always admitted as distinct observations and are accepted,
//     downweighted or rejected later by the optimizer's chi-square gate,
//     so a second sighting of the same place is never silently dropped.
//   - For odometry edges, the first accepted edge on the unordered pair is
//     the incumbent. Compatible duplicates (chi2 <= threshold) are fused
//     information-weighted. Conflicting duplicates are resolved by larger
//     information strength, ties broken by lexicographic edge ID, making the
//     outcome independent of submission order.
func (g *Graph) installEdge(e *Edge) {
	key := pairKey(e.From, e.To)

	if e.Kind == KindLoop {
		e.state = EdgeAccepted
		e.weight = 1
		g.edges[e.ID] = e
		g.dupIndex[key] = append(g.dupIndex[key], e.ID)
		return
	}

	existing := g.dupIndex[key]
	// Odometry: find the incumbent accepted odometry edge on this pair.
	var active *Edge
	var activeID string
	for _, id := range existing {
		cand := g.edges[id]
		if cand != nil && cand.state == EdgeAccepted && cand.Kind == KindOdometry {
			active = cand
			activeID = id
			break
		}
	}
	if active == nil {
		// First odometry link on the pair (loop edges may already exist).
		e.state = EdgeAccepted
		e.writableWeight(1)
		g.edges[e.ID] = e
		g.dupIndex[key] = append(g.dupIndex[key], e.ID)
		return
	}

	chi2 := measurementDisagreement(active, e)
	if chi2 <= g.dupChi2Threshold {
		// Compatible duplicate: information-weighted tangent-space fusion,
		// preserving the orientation of the surviving (first-seen) edge.
		active.Measurement, active.Information = fuseOriented(active, e)
		e.state = EdgeFused
		e.fusedInto = activeID
		e.reason = fmt.Sprintf("duplicate fused chi2=%.4g", chi2)
		active.reason = fmt.Sprintf("fused with %s chi2=%.4g", e.ID, chi2)
		active.writableWeight(1)
	} else {
		// Conflict: strongest information wins, tie by lexicographic edge ID.
		winNew := infoStrength(e) > infoStrength(active)
		if infoStrength(e) == infoStrength(active) && e.ID < activeID {
			winNew = true
		}
		if winNew {
			active.state = EdgeRejected
			active.reason = fmt.Sprintf("冲突被 %s 取代 chi2=%.4g", e.ID, chi2)
			e.state = EdgeAccepted
			e.writableWeight(1)
		} else {
			e.state = EdgeRejected
			e.reason = fmt.Sprintf("conflict with %s chi2=%.4g", activeID, chi2)
		}
	}
	g.edges[e.ID] = e
	g.dupIndex[key] = append(g.dupIndex[key], e.ID)
}

// Node returns a copy of the node record.
// Node returns a copy of the node record.
func (g *Graph) Node(id string) (Node, bool) {
	nd, ok := g.nodes[id]
	if !ok {
		return Node{}, false
	}
	return nd.Node, true
}

// Edge returns a copy of the edge record.
func (g *Graph) Edge(id string) (Edge, bool) {
	if e, ok := g.edges[id]; ok {
		return *e, true
	}
	if e, ok := g.pending[id]; ok {
		return *e, true
	}
	return Edge{}, false
}

// NodeIDs returns node IDs sorted lexicographically.
func (g *Graph) NodeIDs() []string {
	ids := make([]string, 0, len(g.nodes))
	for id := range g.nodes {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// EdgeIDs returns all (including pending/fused/rejected) edge IDs sorted.
func (g *Graph) EdgeIDs() []string {
	ids := make([]string, 0, len(g.edges)+len(g.pending))
	for id := range g.edges {
		ids = append(ids, id)
	}
	for id := range g.pending {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// activeEdges returns accepted/downweighted edges in sorted-ID order.
func (g *Graph) activeEdges() []*Edge {
	var out []*Edge
	for _, e := range g.edges {
		if e.state == EdgeAccepted || e.state == EdgeDownweighted {
			out = append(out, e)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// clone returns a deep copy (used to build immutable versions).
func (g *Graph) clone() *Graph {
	c := NewGraph()
	for id, nd := range g.nodes {
		c.nodes[id] = &nodeData{Node: nd.Node}
	}
	for id, e := range g.edges {
		ec := *e
		c.edges[id] = &ec
	}
	for id, e := range g.pending {
		ec := *e
		c.pending[id] = &ec
	}
	for k, v := range g.dupIndex {
		c.dupIndex[k] = append([]string(nil), v...)
	}
	c.nextSeq = g.nextSeq
	return c
}

// fuseOriented fuses two constraints on the same pair while preserving the
// orientation (From/To) of the active edge.
func fuseOriented(active, e *Edge) (Pose, Mat3) {
	// Both measurements in canonical (lo-hi) orientation.
	_, _, z1, i1 := canonicalMeasurement(active.Constraint)
	_, _, z2, i2 := canonicalMeasurement(e.Constraint)
	z, info := mergeMeasurements([]Pose{z1, z2}, []Mat3{i1, i2}, []float64{1, 1})
	// Re-orient to active's submission direction.
	if active.From < active.To {
		return z, info
	}
	return Inverse(z), info
}
