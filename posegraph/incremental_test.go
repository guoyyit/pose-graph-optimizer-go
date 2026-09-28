package posegraph

import (
	"math"
	"testing"
)

// buildLoopGraph submits a 6-node ring with a loop closure. The same Change set
// is fed to two stores in different slice orders; estimates must match.
func ringChanges(conflict bool) []Change {
	poses := []Pose2{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, math.Pi / 2},
		{1, 1, math.Pi / 2}, {0, 1, math.Pi / 2}}
	var ch []Change
	for i, p := range poses {
		n := Node{ID: NodeID(byte('A' + i)), Initial: p}
		if i == 0 {
			n.Fixed = true
		}
		ch = append(ch, Change{Kind: ChangeNode, Node: n})
	}
	for i := 0; i+1 < len(poses); i++ {
		a := NodeID(byte('A' + i))
		b := NodeID(byte('A' + i + 1))
		ch = append(ch, Change{Kind: ChangeConstraint, Constraint: Constraint{
			ID: "e" + string(byte('a'+i)), A: a, B: b,
			Measurement: between(poses[i], poses[i+1]),
			Covariance:  diagCov(0.01, 0.01, 0.005),
		}})
	}
	// loop F -> A: actual relative pose
	loop := Constraint{ID: "loopFA", A: "F", B: "A", Loop: true,
		Measurement: between(poses[5], poses[0]),
		Covariance:  diagCov(0.02, 0.02, 0.01)}
	ch = append(ch, Change{Kind: ChangeConstraint, Constraint: loop})
	if conflict {
		bad := loop
		bad.Measurement.X += 0.05
		bad.Measurement.Y -= 0.04
		ch = append(ch, Change{Kind: ChangeConstraint, Constraint: bad})
	}
	return ch
}

func reverse(ch []Change) []Change {
	out := append([]Change(nil), ch...)
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

func posesOf(r *Result) map[NodeID]Pose2 {
	m := map[NodeID]Pose2{}
	for k, v := range r.Nodes {
		m[k] = v.Pose
	}
	return m
}

func TestOrderIndependence(t *testing.T) {
	ch := ringChanges(false)
	s1, _ := NewStore(DefaultConfig())
	s2, _ := NewStore(DefaultConfig())
	if _, err := s1.Commit(ch); err != nil {
		t.Fatal(err)
	}
	if _, err := s2.Commit(reverse(ch)); err != nil {
		t.Fatal(err)
	}
	r1, r2 := s1.Query(), s2.Query()
	if r1 == nil || r2 == nil {
		t.Fatal("missing results")
	}
	p1, p2 := posesOf(r1), posesOf(r2)
	for id, a := range p1 {
		b := p2[id]
		if math.Abs(a.X-b.X) > 1e-8 || math.Abs(a.Y-b.Y) > 1e-8 ||
			math.Abs(normalizeAngle(a.Theta-b.Theta)) > 1e-8 {
			t.Errorf("node %s differs by order: %+v vs %+v", id, a, b)
		}
	}
	if math.Abs(r1.Diag.FinalChi2-r2.Diag.FinalChi2) > 1e-10 {
		t.Errorf("chi2 differs: %.12f vs %.12f", r1.Diag.FinalChi2, r2.Diag.FinalChi2)
	}
}

func TestIncrementalVsBatch(t *testing.T) {
	ch := ringChanges(false)
	sb, _ := NewStore(DefaultConfig())
	if _, err := sb.Commit(ch); err != nil {
		t.Fatal(err)
	}
	si, _ := NewStore(DefaultConfig())
	// interleave nodes/constraints in a shuffled, "dangling" order
	order := []int{6, 0, 7, 2, 1, 8, 4, 3, 9, 5, 10, 11}
	for _, k := range order {
		if _, err := si.Submit(ch[k]); err != nil {
			t.Fatal(err)
		}
	}
	rb, ri := sb.Query(), si.Query()
	if rb == nil || ri == nil {
		t.Fatal("missing result")
	}
	if ri.Diag.Status != StatusConverged {
		t.Fatalf("incremental status %s: %v", ri.Diag.Status, ri.Diag.Warnings)
	}
	pb, pi := posesOf(rb), posesOf(ri)
	for id, a := range pb {
		b := pi[id]
		if math.Abs(a.X-b.X) > 1e-7 || math.Abs(a.Y-b.Y) > 1e-7 {
			t.Errorf("node %s batch %+v vs incremental %+v", id, a, b)
		}
	}
}

func TestDanglingPending(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	// constraint before its nodes
	c := Constraint{ID: "ab", A: "A", B: "B", Measurement: Pose2{1, 0, 0},
		Covariance: diagCov(0.01, 0.01, 0.01)}
	rep, err := s.Submit(Change{Kind: ChangeConstraint, Constraint: c})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Reports[0].Outcome != OutcomePending {
		t.Fatalf("want pending, got %s", rep.Reports[0].Outcome)
	}
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "A", Initial: Pose2{}, Fixed: true}})
	rep2, _ := s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "B", Initial: Pose2{2, 0, 0}}})
	found := false
	for _, r := range rep2.Reports {
		if r.Change.Constraint.ID == "ab" && r.Outcome == OutcomeAdded {
			found = true
		}
	}
	if !found {
		t.Fatalf("dangling constraint not replayed: %+v", rep2.Reports)
	}
}

func TestConflictFusionSymmetric(t *testing.T) {
	a := Constraint{ID: "c", A: "X", B: "Y", Measurement: Pose2{1, 0, 0},
		Covariance: diagCov(0.01, 0.01, 0.01)}
	b := a
	b.Measurement = Pose2{1.2, 0.1, 0.05}
	b.Covariance = diagCov(0.04, 0.04, 0.02)
	f1 := fuseConflicts(a, b, 4.0)
	f2 := fuseConflicts(b, a, 4.0)
	for i := 0; i < 9; i++ {
		if math.Abs(f1.Covariance[i]-f2.Covariance[i]) > 1e-12 {
			t.Fatalf("fusion not symmetric at %d: %v vs %v", i, f1.Covariance, f2.Covariance)
		}
	}
	if math.Abs(f1.Measurement.X-f2.Measurement.X) > 1e-12 ||
		math.Abs(f1.Measurement.Y-f2.Measurement.Y) > 1e-12 {
		t.Fatal("fused measurement not symmetric")
	}
	// fused mean must lie between the two measurements
	if f1.Measurement.X < 1.0 || f1.Measurement.X > 1.2 {
		t.Fatalf("fused mean out of bracket: %v", f1.Measurement)
	}
	// fused covariance must reflect both disagreement inflation and precision
	if f1.Covariance[0] <= a.Covariance[0] {
		t.Fatalf("fused covariance not inflated: %v", f1.Covariance)
	}
}

func TestDuplicateIgnored(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "A", Initial: Pose2{}, Fixed: true}})
	c := Constraint{ID: "c", A: "A", B: "A"}
	_ = c
	n := Node{ID: "A", Initial: Pose2{}}
	r1, _ := s.Submit(Change{Kind: ChangeNode, Node: n})
	if r1.Reports[0].Outcome != OutcomeRepeated {
		t.Fatalf("want repeated, got %s", r1.Reports[0].Outcome)
	}
}
