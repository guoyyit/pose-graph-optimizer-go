package posegraph

import (
	"math"
	"testing"
)

func diagCov(x, y, th float64) Mat3 {
	return Mat3{x, 0, 0, 0, y, 0, 0, 0, th}
}

// Build a square-ish chain n0..n4 with noisy initial guesses and exact relative
// measurements; the optimizer should recover measurement-consistent poses.
func TestSmokeChain(t *testing.T) {
	cfg := DefaultConfig()
	s, err := NewStore(cfg)
	if err != nil {
		t.Fatal(err)
	}
	truth := []Pose2{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, math.Pi / 2}, {1, 1, math.Pi}}
	for i, p := range truth {
		// noisy initial guess (the fixed anchor keeps its true origin pose)
		noise := p
		if i > 0 {
			noise.X += 0.15 * float64(i)
			noise.Y -= 0.1 * float64(i-1)
		}
		n := Node{ID: NodeID(string(rune('0' + i))), Initial: noise}
		if i == 0 {
			n.Fixed = true
		}
		if _, err := s.Submit(Change{Kind: ChangeNode, Node: n}); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i+1 < len(truth); i++ {
		a := NodeID(string(rune('0' + i)))
		b := NodeID(string(rune('0' + i + 1)))
		m := between(truth[i], truth[i+1])
		c := Constraint{
			ID: string(rune('a' + i)), A: a, B: b, Measurement: m,
			Covariance: diagCov(0.01, 0.01, 0.01),
		}
		if _, err := s.Submit(Change{Kind: ChangeConstraint, Constraint: c}); err != nil {
			t.Fatal(err)
		}
	}
	res := s.Query()
	if res == nil {
		t.Fatal("expected result")
	}
	if res.Diag.Status != StatusConverged {
		t.Fatalf("status=%s warnings=%v chi2=%.6f", res.Diag.Status, res.Diag.Warnings, res.Diag.FinalChi2)
	}
	if res.Diag.FinalChi2 > 1e-6 {
		t.Fatalf("chi2 too high: %.6f", res.Diag.FinalChi2)
	}
	for i, want := range truth {
		id := NodeID(string(rune('0' + i)))
		got := res.Nodes[id].Pose
		if math.Abs(got.X-want.X) > 1e-4 || math.Abs(got.Y-want.Y) > 1e-4 ||
			math.Abs(normalizeAngle(got.Theta-want.Theta)) > 1e-4 {
			t.Errorf("node %s got %+v want %+v", id, got, want)
		}
	}
}
