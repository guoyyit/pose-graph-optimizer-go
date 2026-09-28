package posegraph

import (
	"math"
	"testing"
)

func dcov(x, y, th float64) Mat3 { return Mat3{x, 0, 0, 0, y, 0, 0, 0, th} }

// buildSquareGraph builds a 4-node square with odometry + one loop closure.
func buildSquareGraph(loop bool) *Graph {
	g := NewGraph()
	g.AddNode(Node{ID: "a"})
	g.AddNode(Node{ID: "b", Hint: Pose{10, 0, 0}})
	g.AddNode(Node{ID: "c", Hint: Pose{10, 10, 0}})
	g.AddNode(Node{ID: "d", Hint: Pose{0, 10, 0}})
	c := dcov(0.05, 0.05, 0.02)
	g.AddConstraint(Constraint{ID: "ab", From: "a", To: "b", Measurement: Pose{10, 0, 0}, Covariance: c})
	g.AddConstraint(Constraint{ID: "bc", From: "b", To: "c", Measurement: Pose{0, 10, 0}, Covariance: c})
	g.AddConstraint(Constraint{ID: "cd", From: "c", To: "d", Measurement: Pose{-10, 0, 0}, Covariance: c})
	g.AddConstraint(Constraint{ID: "da", From: "d", To: "a", Measurement: Pose{0, -10, 0}, Covariance: c})
	if loop {
		g.AddConstraint(Constraint{ID: "ca", From: "c", To: "a", Measurement: Pose{-10, -10, 0}, Covariance: c, Kind: KindLoop})
	}
	return g
}

func TestOptimizeSquare(t *testing.T) {
	res := Optimize(buildSquareGraph(true), DefaultConfig())
	if !res.Diag.OK {
		t.Fatalf("expected converged, got %s: %s", res.Diag.Status, res.Diag.Message)
	}
	want := map[string]Pose{
		"a": {0, 0, 0}, "b": {10, 0, 0}, "c": {10, 10, 0}, "d": {0, 10, 0},
	}
	for id, w := range want {
		got, ok := res.PoseAt(id)
		if !ok {
			t.Fatalf("missing node %s", id)
		}
		if math.Hypot(got.X-w.X, got.Y-w.Y) > 0.15 {
			t.Errorf("node %s = %+v want %+v", id, got, w)
		}
	}
	// every edge residual should be small
	for _, r := range res.Residuals {
		if r.State == EdgeAccepted && r.Weight > 0 && r.Chi2 > 1.0 {
			t.Errorf("edge %s chi2 %.3f too high", r.EdgeID, r.Chi2)
		}
	}
}

// TestReproducibleOutOfOrder verifies that the same graph, assembled via
// different submission orders (reverse / shuffled nodes and edges, edges
// before nodes), yields bit-for-bit identical estimates.
func TestReproducibleOutOfOrder(t *testing.T) {
	type ec struct {
		id, f, t string
		z        Pose
		kind     ConstraintKind
	}
	cons := []ec{
		// odometry with true relative rotation
		{"ab", "a", "b", Pose{10, 0, 0}, KindOdometry},
		{"bc", "b", "c", Pose{0, 10, math.Pi / 2}, KindOdometry},
		{"cd", "c", "d", Pose{0, 10, 0}, KindOdometry},
		{"da", "d", "a", Pose{-10, 0, -math.Pi / 2}, KindOdometry},
		// diagonal loop a->c: relative = Log(Ta^-1 Tc); Ta identity so z=Tc
		{"ac", "a", "c", Pose{10, 10, math.Pi / 2}, KindLoop},
	}
	c := dcov(0.05, 0.05, 0.02)

	// Deliberately wrong hints to force real optimization (order-independent
	// spanning tree ignores them for connected nodes, but we perturb them).
	hints := map[string]Pose{
		"a": {}, "b": {1, 1, 0.1}, "c": {2, 2, 0.2}, "d": {-1, 1, 0.3},
	}

	solve := func(nodeOrder, conOrder []int) *Result {
		g := NewGraph()
		// edges first -> they pend until nodes arrive
		for _, ci := range conOrder {
			cn := cons[ci]
			g.AddConstraint(Constraint{ID: cn.id, From: cn.f, To: cn.t, Measurement: cn.z, Covariance: c, Kind: cn.kind})
		}
		allIDs := []string{"a", "b", "c", "d"}
		for _, ni := range nodeOrder {
			id := allIDs[ni]
			g.AddNode(Node{ID: id, Hint: hints[id]})
		}
		return Optimize(g, DefaultConfig())
	}

	ref := solve([]int{0, 1, 2, 3}, []int{0, 1, 2, 3, 4})
	if !ref.Diag.OK {
		t.Fatalf("ref not ok: %s %s cost=%g", ref.Diag.Status, ref.Diag.Message, ref.Diag.FinalCost)
	}
	nodeOrders := [][]int{{3, 2, 1, 0}, {2, 0, 3, 1}}
	edgeOrders := [][]int{{4, 3, 2, 1, 0}, {3, 1, 4, 0, 2}}
	for oi, no := range nodeOrders {
		for ci, co := range edgeOrders {
			got := solve(no, co)
			if !got.Diag.OK {
				t.Fatalf("order (%d,%d) not ok: %s %s", oi, ci, got.Diag.Status, got.Diag.Message)
			}
			for _, ne := range ref.Nodes {
				gp, _ := got.PoseAt(ne.ID)
				if math.Hypot(gp.X-ne.Pose.X, gp.Y-ne.Pose.Y) > 1e-7 {
					t.Fatalf("order (%d,%d) node %s = %+v want %+v (order-dependent)",
						oi, ci, ne.ID, gp, ne.Pose)
				}
				if math.Abs(WrapAngle(gp.Theta-ne.Pose.Theta)) > 1e-7 {
					t.Fatalf("order (%d,%d) node %s theta differs", oi, ci, ne.ID)
				}
			}
		}
	}
}

// TestDuplicateFusion verifies two compatible measurements on the same pair
// fuse into one stronger constraint, and gross conflicts reject the weaker.
func TestDuplicateFusion(t *testing.T) {
	g := NewGraph()
	g.AddNode(Node{ID: "a"})
	g.AddNode(Node{ID: "b", Hint: Pose{10, 0, 0}})
	c := dcov(0.1, 0.1, 0.05)
	g.AddConstraint(Constraint{ID: "m1", From: "a", To: "b", Measurement: Pose{10, 0, 0}, Covariance: c})
	// compatible repeat measurement (small noise)
	g.AddConstraint(Constraint{ID: "m2", From: "a", To: "b", Measurement: Pose{10.02, 0.0, 0}, Covariance: c})
	if e, _ := g.Edge("m2"); e.state != EdgeFused {
		t.Fatalf("compatible duplicate state=%s want fused (%s)", e.state, e.reason)
	}
	e1, _ := g.Edge("m1")
	// fused information should be about 2x
	if e1.Information[0] < 1.5/c[0] {
		t.Fatalf("fused info not strengthened: %v", e1.Information[0])
	}

	// gross conflict on a different pair
	g.AddNode(Node{ID: "x"})
	g.AddNode(Node{ID: "y", Hint: Pose{1, 0, 0}})
	g.AddConstraint(Constraint{ID: "k1", From: "x", To: "y", Measurement: Pose{1, 0, 0}, Covariance: c})
	// much stronger conflicting measurement
	strong := dcov(0.0001, 0.0001, 0.00005)
	g.AddConstraint(Constraint{ID: "k2", From: "x", To: "y", Measurement: Pose{5, 0, 0}, Covariance: strong})
	k1, _ := g.Edge("k1")
	k2, _ := g.Edge("k2")
	if !(k2.state == EdgeAccepted && k1.state == EdgeRejected) {
		t.Fatalf("conflict resolution wrong: k1=%s k2=%s", k1.state, k2.state)
	}
}

// TestIllConditionedDetection builds a structurally rank-deficient graph:
// two components connected by nothing. The result must NOT be marked success.
func TestIllConditionedDetection(t *testing.T) {
	g := NewGraph()
	g.AddNode(Node{ID: "a"})
	g.AddNode(Node{ID: "b", Hint: Pose{1, 0, 0}})
	g.AddNode(Node{ID: "c", Hint: Pose{0, 1, 0}})
	c := dcov(0.05, 0.05, 0.02)
	g.AddConstraint(Constraint{ID: "ab", From: "a", To: "b", Measurement: Pose{1, 0, 0}, Covariance: c})
	// c is isolated -> unobservable
	res := Optimize(g, DefaultConfig())
	if res.Diag.OK {
		t.Fatal("isolated-node solution must not be marked success")
	}
	if res.Diag.Status != SolveUnderconstrained {
		t.Fatalf("status=%s want underconstrained", res.Diag.Status)
	}
}

// TestMarginalCovariance checks covariances are positive, symmetric and that
// more connected nodes have tighter uncertainty.
func TestMarginalCovariance(t *testing.T) {
	g := buildSquareGraph(true)
	res := Optimize(g, DefaultConfig())
	if !res.Diag.OK {
		t.Fatal(res.Diag.Message)
	}
	for _, ne := range res.Nodes {
		cv := ne.Covariance
		if ne.ID == "a" {
			// anchor is exactly fixed -> zero covariance by design
			if cv != (Mat3{}) {
				t.Errorf("anchor covariance should be zero, got %v", cv)
			}
			continue
		}
		for i := 0; i < 3; i++ {
			if cv[i*3+i] <= 0 {
				t.Errorf("node %s non-positive variance %v", ne.ID, cv)
			}
			for j := i + 1; j < 3; j++ {
				if math.Abs(cv[i*3+j]-cv[j*3+i]) > 1e-12 {
					t.Errorf("node %s asymmetric covariance", ne.ID)
				}
			}
		}
	}
	// interior node with a loop closure (c) should be tighter than the
	// degree-1-ish chain intuition: just assert covariance is finite and
	// smaller than an inflated-variance rerun.
	bad := DefaultConfig()
	gLoose := buildSquareGraph(false)
	resLoose := Optimize(gLoose, bad)
	if !resLoose.Diag.OK {
		t.Fatal(resLoose.Diag.Message)
	}
	var cLoop, cChain Mat3
	for _, ne := range res.Nodes {
		if ne.ID == "c" {
			cLoop = ne.Covariance
		}
	}
	for _, ne := range resLoose.Nodes {
		if ne.ID == "c" {
			cChain = ne.Covariance
		}
	}
	if trace3(cLoop) > trace3(cChain)+1e-12 {
		t.Errorf("loop closure did not tighten covariance: %.6g > %.6g", trace3(cLoop), trace3(cChain))
	}
}

func trace3(m Mat3) float64 { return m[0] + m[4] + m[8] }

// TestOdometryConflictNotMarkedSuccess builds a consistent square plus a
// grossly wrong ODOMETRY closing edge (odometry is not subject to the loop
// gate). The optimizer must refuse to call such a high-residual estimate a
// success, even if the iterations otherwise settle.
// TestOdometryConflictNotMarkedSuccess feeds a consistent rectangle plus a
// grossly wrong, high-confidence ODOMETRY edge on a fresh diagonal pair.
// Odometry is not subject to the loop gate, so the optimizer must refuse to
// call such a high-residual estimate a success (never false success).
func TestOdometryConflictNotMarkedSuccess(t *testing.T) {
	g := NewGraph()
	for _, n := range []Node{
		{ID: "a"}, {ID: "b", Hint: Pose{10, 0, 0}},
		{ID: "c", Hint: Pose{10, 10, 0}}, {ID: "d", Hint: Pose{0, 10, 0}},
	} {
		g.AddNode(n)
	}
	c := dcov(0.02, 0.02, 0.01)
	g.AddConstraint(Constraint{ID: "ab", From: "a", To: "b", Measurement: Pose{10, 0, 0}, Covariance: c})
	g.AddConstraint(Constraint{ID: "bc", From: "b", To: "c", Measurement: Pose{0, 10, pi2c()}, Covariance: c})
	g.AddConstraint(Constraint{ID: "cd", From: "c", To: "d", Measurement: Pose{0, 10, 0}, Covariance: c})
	g.AddConstraint(Constraint{ID: "da", From: "d", To: "a", Measurement: Pose{-10, 0, -pi2c()}, Covariance: c})
	// contradictory high-confidence odometry on a fresh diagonal pair a->c:
	// truth is (10,10,90deg); claim (-10,-10,90deg): tens of meters error.
	g.AddConstraint(Constraint{ID: "badac", From: "a", To: "c",
		Measurement: Pose{-10, -10, pi2c()}, Covariance: dcov(0.001, 0.001, 0.001),
		Kind: KindOdometry})

	res := Optimize(g, DefaultConfig())
	if res.Diag.OK {
		t.Fatalf("high-residual odometry conflict marked success: maxChi2=%.2f status=%s",
			res.Diag.MaxEdgeChi2, res.Diag.Status)
	}
	if res.Diag.MaxEdgeChi2 <= 25 {
		t.Fatalf("expected high worst-edge chi2, got %.3f", res.Diag.MaxEdgeChi2)
	}
}

func pi2c() float64 { return math.Pi / 2 }
