package posegraph

import "testing"

func TestDisconnectedIsIllConditioned(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	// anchored component A-B
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "A", Initial: Pose2{}, Fixed: true}})
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "B", Initial: Pose2{1, 0, 0}}})
	s.Submit(Change{Kind: ChangeConstraint, Constraint: Constraint{
		ID: "ab", A: "A", B: "B", Measurement: Pose2{1, 0, 0},
		Covariance: diagCov(0.01, 0.01, 0.01)}})
	// isolated free node C with no edges and no prior
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "C", Initial: Pose2{5, 5, 0}}})
	// explicit optimization of the now-singular graph must diagnose ill-conditioning
	diag, err := s.Optimize()
	if err != nil {
		t.Fatal(err)
	}
	if diag.Status != StatusIllConditioned {
		t.Fatalf("want ill_conditioned, got %s: %v", diag.Status, diag.Warnings)
	}
	// the prior consistent result for the healthy component remains queryable
	if s.Query() == nil {
		t.Fatal("previous consistent result should be retained after failed solve")
	}
}

func TestDegenerateConstraintCovariance(t *testing.T) {
	// zero covariance must not become infinite information; floor applies.
	cfg := DefaultConfig()
	g := newGraph()
	g.nodes["A"] = Node{ID: "A", Initial: Pose2{}, Fixed: true}
	g.nodes["B"] = Node{ID: "B", Initial: Pose2{1, 0, 0}}
	g.constraints["ab"] = Constraint{ID: "ab", A: "A", B: "B",
		Measurement: Pose2{1, 0, 0}, Covariance: Mat3{}}
	out := runOptimization(cfg, g, nil)
	if out.diag.Status == StatusFailed || out.diag.Status == StatusIllConditioned {
		t.Fatalf("floored covariance should still solve: %s %v", out.diag.Status, out.diag.Warnings)
	}
}

func TestPriorRemovesAnchorNeed(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	// no fixed node, but a prior anchors the gauge
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "A", Initial: Pose2{0.1, 0.1, 0.05}}})
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "B", Initial: Pose2{1.2, 0.1, 0.05}}})
	s.Submit(Change{Kind: ChangeConstraint, Constraint: Constraint{
		ID: "pA", Kind: Prior, B: "A", Measurement: Pose2{0, 0, 0},
		Covariance: diagCov(0.001, 0.001, 0.001)}})
	s.Submit(Change{Kind: ChangeConstraint, Constraint: Constraint{
		ID: "ab", A: "A", B: "B", Measurement: Pose2{1, 0, 0},
		Covariance: diagCov(0.01, 0.01, 0.01)}})
	res := s.Query()
	if res == nil || res.Diag.Status != StatusConverged {
		t.Fatalf("prior-anchored graph should converge: %+v", res)
	}
	a := res.Nodes["A"].Pose
	if a.X > 0.01 {
		t.Fatalf("prior should pull A near origin: %+v", a)
	}
}

func TestMarginalsShrinkWithMoreConstraints(t *testing.T) {
	cfg := DefaultConfig()
	g := newGraph()
	g.nodes["A"] = Node{ID: "A", Initial: Pose2{}, Fixed: true}
	g.nodes["B"] = Node{ID: "B", Initial: Pose2{1, 0, 0}}
	g.constraints["ab"] = Constraint{ID: "ab", A: "A", B: "B",
		Measurement: Pose2{1, 0, 0}, Covariance: diagCov(0.1, 0.1, 0.1)}
	o1 := runOptimization(cfg, g, nil)
	g.constraints["ab2"] = Constraint{ID: "ab2", A: "A", B: "B",
		Measurement: Pose2{1, 0, 0}, Covariance: diagCov(0.1, 0.1, 0.1)}
	o2 := runOptimization(cfg, g, nil)
	if !o1.ok || !o2.ok {
		t.Fatal("both runs should solve")
	}
	if o2.marginals["B"].Covariance[0] >= o1.marginals["B"].Covariance[0] {
		t.Fatalf("more information must reduce variance: %v -> %v",
			o1.marginals["B"].Covariance[0], o2.marginals["B"].Covariance[0])
	}
}

func TestStagnationNotReportedSuccess(t *testing.T) {
	// A contradictory constraint with tiny covariance that cannot be satisfied:
	// solver must not return a high-error result marked successful.
	cfg := DefaultConfig()
	g := newGraph()
	g.nodes["A"] = Node{ID: "A", Initial: Pose2{}, Fixed: true}
	g.nodes["B"] = Node{ID: "B", Initial: Pose2{1, 0, 0}}
	g.constraints["ab"] = Constraint{ID: "ab", A: "A", B: "B",
		Measurement: Pose2{1, 0, 0}, Covariance: diagCov(1e-8, 1e-8, 1e-8)}
	g.constraints["ab2"] = Constraint{ID: "ab2", A: "A", B: "B",
		Measurement: Pose2{1.5, 0, 0}, Covariance: diagCov(1e-8, 1e-8, 1e-8)}
	out := runOptimization(cfg, g, nil)
	// Two contradictory hard constraints produce a large unavoidable residual.
	// The run must not be flagged as an acceptable (publishable) result even
	// though the least-squares iterate itself converges.
	if out.ok {
		t.Fatalf("high-residual compromise must not be marked ok: status=%s maxW=%.2f",
			out.diag.Status, out.diag.MaxWhitened)
	}
}
