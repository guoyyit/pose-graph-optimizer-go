package posegraph

import (
	"math"
	"testing"
)

const piHalf = math.Pi / 2

// TestGateBands verifies the hard/soft/accept bands directly on the solver.
func TestGateBands(t *testing.T) {
	gateLo := chiSquareThreshold(3, 0.95) // 7.815
	gateHi := chiSquareThreshold(3, 0.99) // 11.345
	cfg := DefaultConfig()

	decide := func(chi2 float64) (weight float64, action EdgeState) {
		// emulate applyGate for a single synthetic loop edge
		fake := &solver{cfg: cfg, gateLo: gateLo, gateHi: gateHi,
			edges:   []*Edge{{Constraint: Constraint{ID: "L", Kind: KindLoop}}},
			weights: []float64{1}}
		d := &Diagnostics{}
		fake.applyGate([]float64{chi2}, d)
		if len(d.Rejected) > 0 {
			return fake.weights[0], EdgeRejected
		}
		if len(d.Downweighted) > 0 {
			return fake.weights[0], EdgeDownweighted
		}
		return fake.weights[0], EdgeAccepted
	}

	// well inside: accept at full weight
	if w, a := decide(2.0); a != EdgeAccepted || w != 1 {
		t.Fatalf("chi2=2 -> %v w=%v want accept/1", a, w)
	}
	// soft band: downweighted, still positive
	w, a := decide(9.0)
	if a != EdgeDownweighted || w <= 0 || w >= 1 {
		t.Fatalf("chi2=9 -> %v w=%v want downweight in (0,1)", a, w)
	}
	// hard outlier: rejected at zero weight
	if w, a := decide(50.0); a != EdgeRejected || w != 0 {
		t.Fatalf("chi2=50 -> %v w=%v want reject/0", a, w)
	}
}

// TestSoftLoopAcceptedEndToEnd verifies a mildly inconsistent loop that stays
// below the hard gate does not destroy the estimate; the solution converges
// and the loop carries non-negative weight.
func TestSoftLoopAcceptedEndToEnd(t *testing.T) {
	g := NewGraph()
	for _, n := range []Node{
		{ID: "a"}, {ID: "b", Hint: Pose{10, 0, 0}},
		{ID: "c", Hint: Pose{10, 10, 0}}, {ID: "d", Hint: Pose{0, 10, 0}},
	} {
		g.AddNode(n)
	}
	c := dcov(0.05, 0.05, 0.02)
	g.AddConstraint(Constraint{ID: "ab", From: "a", To: "b", Measurement: Pose{10, 0, 0}, Covariance: c})
	g.AddConstraint(Constraint{ID: "bc", From: "b", To: "c", Measurement: Pose{0, 10, piHalf}, Covariance: c})
	g.AddConstraint(Constraint{ID: "cd", From: "c", To: "d", Measurement: Pose{0, 10, 0}, Covariance: c})
	g.AddConstraint(Constraint{ID: "da", From: "d", To: "a", Measurement: Pose{-10, 0, -piHalf}, Covariance: c})
	// small ~0.2m error: below the hard gate, the estimate stays globally good
	g.AddConstraint(Constraint{ID: "soft", From: "c", To: "a",
		Measurement: Pose{-10.2, 10.2, -piHalf}, Covariance: c, Kind: KindLoop})
	res := Optimize(g, DefaultConfig())
	if !res.Diag.OK {
		t.Fatalf("soft mismatch should solve: %s %s", res.Diag.Status, res.Diag.Message)
	}
	var soft *Residual
	for i := range res.Residuals {
		if res.Residuals[i].EdgeID == "soft" {
			soft = &res.Residuals[i]
		}
	}
	if soft == nil || soft.Weight < 0 {
		t.Fatal("soft edge missing or negative weight")
	}
	if soft.Weight == 0 {
		t.Fatalf("mild loop hard-rejected; chi2=%g", soft.Chi2)
	}
	// trajectory remains geometrically correct
	pc, _ := res.PoseAt("c")
	if d := math.Hypot(pc.X-10, pc.Y-10); d > 1.0 {
		t.Fatalf("trajectory distorted by mild loop: c=%+v dist=%.3f", pc, d)
	}
}

func TestHuberWeights(t *testing.T) {
	// residual magnitude 2 with delta 4 => weight 1
	w := huberWeights([]float64{4}, 4) // sqrt=2
	if w[0] != 1 {
		t.Fatalf("inlier weight=%v want 1", w[0])
	}
	// magnitude 10 with delta 4 => 4/10 = 0.4
	w = huberWeights([]float64{100}, 4)
	if d := abs(w[0] - 0.4); d > 1e-12 {
		t.Fatalf("huber outlier weight=%v want 0.4", w[0])
	}
	// disabled kernel
	w = huberWeights([]float64{1e6}, 0)
	if w[0] != 1 {
		t.Fatalf("disabled kernel weight=%v want 1", w[0])
	}
}

func TestMergeMeasurements(t *testing.T) {
	// two identical measurements => fused pose identical, info doubled
	m := Pose{3, 4, 0.2}
	info := identity3()
	z, fused := mergeMeasurements([]Pose{m, m}, []Mat3{info, info}, []float64{1, 1})
	if abs(z.X-3) > 1e-9 || abs(z.Y-4) > 1e-9 || abs(z.Theta-0.2) > 1e-9 {
		t.Fatalf("fused pose wrong: %+v", z)
	}
	if abs(fused[0]-2) > 1e-9 || abs(fused[4]-2) > 1e-9 {
		t.Fatalf("information not doubled: %v", fused)
	}
}

func TestDiagnosticsFields(t *testing.T) {
	g := NewGraph()
	g.AddNode(Node{ID: "a"})
	g.AddNode(Node{ID: "b", Hint: Pose{5, 0, 0}})
	g.AddConstraint(Constraint{From: "a", To: "b", Measurement: Pose{5, 0, 0},
		Covariance: dcov(0.1, 0.1, 0.05)})
	res := Optimize(g, DefaultConfig())
	if !res.Diag.OK || res.Diag.Iterations <= 0 {
		t.Fatal("expected a converged run with iterations")
	}
	if res.Diag.FinalCost < 0 {
		t.Fatal("negative cost")
	}
	if len(res.Residuals) != 1 {
		t.Fatalf("want 1 residual, got %d", len(res.Residuals))
	}
}
