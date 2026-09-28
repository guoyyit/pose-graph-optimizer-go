package posegraph

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Graph is the deterministic container of accepted nodes and constraints.
// All bulk iteration happens in lexicographic ID order so results never depend
// on submission order.
type Graph struct {
	nodes       map[NodeID]Node
	constraints map[string]Constraint
}

// Errors surfaced when validating changes.
var (
	errEmptyNodeID       = errors.New("posegraph: node id is empty")
	errEmptyConstraintID = errors.New("posegraph: constraint id is empty")
	errUnknownEndpoint   = errors.New("posegraph: constraint references unknown node")
	errBadEndpoints      = errors.New("posegraph: relative constraint needs two distinct endpoints")
	errBadPrior          = errors.New("posegraph: prior constraint needs exactly one endpoint (B)")
	errNonFinitePose     = errors.New("posegraph: non-finite pose value")
	errBadCovariance     = errors.New("posegraph: covariance must be symmetric with non-negative diagonal")
)

func newGraph() *Graph {
	return &Graph{nodes: map[NodeID]Node{}, constraints: map[string]Constraint{}}
}

func (g *Graph) nodeCount() int       { return len(g.nodes) }
func (g *Graph) constraintCount() int { return len(g.constraints) }

func (g *Graph) getNode(id NodeID) (Node, bool) {
	n, ok := g.nodes[id]
	return n, ok
}

func (g *Graph) getConstraint(id string) (Constraint, bool) {
	c, ok := g.constraints[id]
	return c, ok
}

// sortedNodeIDs returns lexicographically sorted node IDs.
func (g *Graph) sortedNodeIDs() []NodeID {
	ids := make([]NodeID, 0, len(g.nodes))
	for id := range g.nodes {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// sortedConstraintIDs returns lexicographically sorted constraint IDs.
func (g *Graph) sortedConstraintIDs() []string {
	ids := make([]string, 0, len(g.constraints))
	for id := range g.constraints {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// deepCopy returns an independent copy.
func (g *Graph) deepCopy() *Graph {
	ng := newGraph()
	for id, n := range g.nodes {
		ng.nodes[id] = n
	}
	for id, c := range g.constraints {
		ng.constraints[id] = c
	}
	return ng
}

func poseFinite(p Pose2) bool {
	return isFinite(p.X) && isFinite(p.Y) && isFinite(p.Theta)
}

func validateNode(n Node) error {
	if n.ID == "" {
		return errEmptyNodeID
	}
	if !poseFinite(n.Initial) {
		return errNonFinitePose
	}
	return nil
}

func covarianceValid(m Mat3) bool {
	for i := 0; i < 9; i++ {
		if !isFinite(m[i]) {
			return false
		}
	}
	for i := 0; i < 3; i++ {
		if m[i*3+i] < 0 {
			return false
		}
	}
	// symmetry
	const symTol = 1e-9
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if math.Abs(m[i*3+j]-m[j*3+i]) > symTol*(1+math.Abs(m[i*3+j])) {
				return false
			}
		}
	}
	return true
}

func validateConstraint(c Constraint, known func(NodeID) bool) error {
	if c.ID == "" {
		return errEmptyConstraintID
	}
	if !poseFinite(c.Measurement) {
		return errNonFinitePose
	}
	if !covarianceValid(c.Covariance) {
		return errBadCovariance
	}
	switch c.Kind {
	case Relative:
		if c.A == "" || c.B == "" || c.A == c.B {
			return errBadEndpoints
		}
		if !known(c.A) || !known(c.B) {
			return errUnknownEndpoint
		}
	case Prior:
		if c.B == "" || c.A != "" {
			return errBadPrior
		}
		if !known(c.B) {
			return errUnknownEndpoint
		}
	default:
		return fmt.Errorf("posegraph: unknown constraint kind %d", c.Kind)
	}
	return nil
}

// nodesEqual reports whether a repeated node submission carries the same
// initial pose and fixed flag. Different initial poses are ignored on repeat
// (the first submission wins) but reported as a conflict by the store.
func nodesEqual(a, b Node) bool {
	const tol = 1e-9
	return a.ID == b.ID &&
		a.Fixed == b.Fixed &&
		math.Abs(a.Initial.X-b.Initial.X) <= tol &&
		math.Abs(a.Initial.Y-b.Initial.Y) <= tol &&
		math.Abs(normalizeAngle(a.Initial.Theta-b.Initial.Theta)) <= tol
}

// constraintsCompatible reports whether two submissions with the same ID share
// endpoints and agree on measurement and covariance within tolerance.
func constraintsCompatible(a, b Constraint) bool {
	const tol = 1e-6
	if a.Kind != b.Kind || a.A != b.A || a.B != b.B || a.Loop != b.Loop {
		return false
	}
	dx := a.Measurement.X - b.Measurement.X
	dy := a.Measurement.Y - b.Measurement.Y
	dth := normalizeAngle(a.Measurement.Theta - b.Measurement.Theta)
	if math.Abs(dx) > tol*(1+math.Abs(a.Measurement.X)) ||
		math.Abs(dy) > tol*(1+math.Abs(a.Measurement.Y)) ||
		math.Abs(dth) > tol {
		return false
	}
	for i := 0; i < 9; i++ {
		if math.Abs(a.Covariance[i]-b.Covariance[i]) > tol*(1+math.Abs(a.Covariance[i])) {
			return false
		}
	}
	return true
}

// weightedAngle averages two angles using unit-vector fusion, which is
// symmetric and handles the ±pi wrap without a branchy mean.
func weightedAngle(ta, tb, wa, wb float64) float64 {
	s := wa*math.Sin(ta) + wb*math.Sin(tb)
	c := wa*math.Cos(ta) + wb*math.Cos(tb)
	return normalizeAngle(math.Atan2(s, c))
}

func matAdd(a, b Mat3) Mat3 {
	for i := 0; i < 9; i++ {
		a[i] += b[i]
	}
	return a
}

// fuseConflicts merges two submissions carrying the same constraint ID but
// disagreeing measurement/covariance. The rule is deliberately symmetric:
//
//	Σ_fused = (Σ₁⁻¹ + Σ₂⁻¹)⁻¹ · inflation
//	z_fused = weighted mean of the two measurements with weights tr(Σ⁻¹)
//
// so fusing (a,b) yields exactly the same result as fusing (b,a), and the
// inflated covariance records the disagreement as reduced certainty.
func fuseConflicts(a, b Constraint, inflation float64) Constraint {
	ia, _ := covarianceInformation(a.Covariance, Vec3{})
	ib, _ := covarianceInformation(b.Covariance, Vec3{})
	sum := matAdd(ia, ib)
	cov, ok := invertMat3(sum)
	if !ok { // extremely unlikely after flooring; fall back to larger covariance
		cov = a.Covariance
	}
	cov = matScale(cov, inflation)

	// Weight by total information (trace) — a scalar, hence commutative.
	waRaw := ia[0] + ia[4] + ia[8]
	wbRaw := ib[0] + ib[4] + ib[8]
	if waRaw+wbRaw <= 0 {
		waRaw, wbRaw = 1, 1
	}
	wa := waRaw / (waRaw + wbRaw)
	wb := wbRaw / (waRaw + wbRaw)
	m := Pose2{
		X:     wa*a.Measurement.X + wb*b.Measurement.X,
		Y:     wa*a.Measurement.Y + wb*b.Measurement.Y,
		Theta: weightedAngle(a.Measurement.Theta, b.Measurement.Theta, wa, wb),
	}
	out := a
	out.Measurement = m
	out.Covariance = symmetrize3(cov)
	out.Information = Mat3{}
	out.RobustWeight = 1
	return out
}

// defaultConstraintID builds a deterministic ID from endpoints and a formatted
// measurement, so callers may omit explicit IDs without losing reproducibility.
func defaultConstraintID(c Constraint) string {
	if c.Kind == Prior {
		return fmt.Sprintf("prior:%s:p%.6f,p%.6f,p%.4f", c.B,
			c.Measurement.X, c.Measurement.Y, c.Measurement.Theta)
	}
	if c.A < c.B {
		return fmt.Sprintf("rel:%s:%s:p%.6f,p%.6f,p%.4f", c.A, c.B,
			c.Measurement.X, c.Measurement.Y, c.Measurement.Theta)
	}
	// reversed storage: invert the measurement so semantics stay identical
	inv := inversePose(c.Measurement)
	return fmt.Sprintf("rel:%s:%s:p%.6f,p%.6f,p%.4f", c.B, c.A,
		inv.X, inv.Y, inv.Theta)
}
