package posegraph

import (
	"math"
	"testing"
)

// straightChain builds a 4-node straight line anchored at A.
func straightChain(t *testing.T, s *Store) {
	t.Helper()
	truth := []Pose2{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}}
	for i, p := range truth {
		n := Node{ID: NodeID(byte('A' + i)), Initial: p}
		if i == 0 {
			n.Fixed = true
		}
		if _, err := s.Submit(Change{Kind: ChangeNode, Node: n}); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i+1 < len(truth); i++ {
		c := Constraint{
			ID: string(byte('e' + i)), A: NodeID(byte('A' + i)), B: NodeID(byte('A' + i + 1)),
			Measurement: Pose2{1, 0, 0}, Covariance: diagCov(0.01, 0.01, 0.01),
		}
		if _, err := s.Submit(Change{Kind: ChangeConstraint, Constraint: c}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLoopRejected(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	straightChain(t, s)
	before := s.Query()
	if before == nil {
		t.Fatal("baseline result missing")
	}
	// wildly wrong loop: claims D is 10m off in Y from A
	bad := Constraint{ID: "badloop", A: "D", B: "A", Loop: true,
		Measurement: Pose2{-3, 10, 0}, Covariance: diagCov(0.01, 0.01, 0.01)}
	rep, err := s.Submit(Change{Kind: ChangeConstraint, Constraint: bad})
	if err != nil {
		t.Fatal(err)
	}
	var lr ChangeReport
	for _, r := range rep.Reports {
		if r.Change.Constraint.ID == "badloop" {
			lr = r
		}
	}
	if lr.Outcome != OutcomeRejected {
		t.Fatalf("want rejected, got %s (%s)", lr.Outcome, lr.Detail)
	}
	// rejected constraint must not be in the graph and a rejected list exists
	snap := s.Snapshot()
	if _, present := snap.Constraints["badloop"]; present {
		t.Fatal("rejected loop entered the graph")
	}
	if len(snap.Rejected) != 1 {
		t.Fatalf("want 1 rejected entry, got %d", len(snap.Rejected))
	}
	// the rest of the graph must still estimate correctly
	after := s.Query()
	if after == nil {
		t.Fatal("estimate unavailable after rejection")
	}
	if after.Diag.Status != StatusConverged {
		t.Fatalf("status after reject: %s", after.Diag.Status)
	}
	d := after.Nodes["D"].Pose
	if math.Abs(d.X-3) > 1e-5 || math.Abs(d.Y) > 1e-5 {
		t.Fatalf("graph distorted by rejected loop: %+v", d)
	}
}

func TestLoopDownweighted(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	straightChain(t, s)
	// mildly inconsistent loop (0.3m sideways error with tight covariance):
	// above robust kernel but below the hard chi2 gate.
	mild := Constraint{ID: "mildloop", A: "D", B: "A", Loop: true,
		Measurement: Pose2{-3, 0.35, 0}, Covariance: diagCov(0.01, 0.01, 0.02)}
	rep, err := s.Submit(Change{Kind: ChangeConstraint, Constraint: mild})
	if err != nil {
		t.Fatal(err)
	}
	var lr ChangeReport
	for _, r := range rep.Reports {
		if r.Change.Constraint.ID == "mildloop" {
			lr = r
		}
	}
	if lr.Outcome != OutcomeLoopAccept && lr.Outcome != OutcomeLoopDownweight {
		t.Fatalf("unexpected outcome %s: %s chi2=%.2f", lr.Outcome, lr.Detail, lr.Chi2PerDOF)
	}
	// with 0.35m and sigma 0.1m the chi2/DOF ~ (0.35/0.1)^2/3 ~ 4.1 < 7.815 gate,
	// whitened norm ~3.5 > kernel 1, so it should be down-weighted.
	if lr.Outcome != OutcomeLoopDownweight {
		t.Fatalf("expected downweight, got %s (chi2/DOF=%.2f)", lr.Outcome, lr.Chi2PerDOF)
	}
	if lr.Weight >= 1 || lr.Weight <= 0 {
		t.Fatalf("robust weight out of range: %v", lr.Weight)
	}
}

func TestLoopConsistentAccepted(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	straightChain(t, s)
	good := Constraint{ID: "goodloop", A: "D", B: "A", Loop: true,
		Measurement: Pose2{-3, 0, 0}, Covariance: diagCov(0.02, 0.02, 0.02)}
	rep, _ := s.Submit(Change{Kind: ChangeConstraint, Constraint: good})
	for _, r := range rep.Reports {
		if r.Change.Constraint.ID == "goodloop" && r.Outcome != OutcomeLoopAccept {
			t.Fatalf("want accept, got %s", r.Outcome)
		}
	}
}

func TestRobustWeight(t *testing.T) {
	cfg := DefaultConfig()
	if robustWeight(cfg, 0.5) != 1 {
		t.Fatal("within-delta should be 1")
	}
	w1 := robustWeight(cfg, 2)
	w2 := robustWeight(cfg, 10)
	if w1 >= 1 || w2 >= w1 {
		t.Fatalf("weights should decrease: %v %v", w1, w2)
	}
}
