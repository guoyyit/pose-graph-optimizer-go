package posegraph

import (
	"math"
	"path/filepath"
	"testing"
)

func buildPersistentChain(t *testing.T, path string) *Store {
	cfg := DefaultConfig()
	cfg.SnapshotPath = path
	s, err := NewStore(cfg)
	if err != nil {
		t.Fatal(err)
	}
	truth := []Pose2{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, math.Pi / 2}}
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
			Measurement: between(truth[i], truth[i+1]),
			Covariance:  diagCov(0.01, 0.01, 0.01),
		}
		if _, err := s.Submit(Change{Kind: ChangeConstraint, Constraint: c}); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func TestRestartRecovery(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")

	s1 := buildPersistentChain(t, path)
	r1 := s1.Query()
	if r1 == nil {
		t.Fatal("no result before restart")
	}
	if err := s1.Close(); err != nil {
		t.Fatal(err)
	}

	// reopen: state must be restored, no accepted constraint lost
	cfg := DefaultConfig()
	cfg.SnapshotPath = path
	s2, err := NewStore(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()

	snap := s2.Snapshot()
	if len(snap.Nodes) != 4 {
		t.Fatalf("nodes lost on restart: %d", len(snap.Nodes))
	}
	if len(snap.Constraints) != 3 {
		t.Fatalf("constraints lost on restart: %d", len(snap.Constraints))
	}
	if s2.Version() != s1.Version() {
		t.Fatalf("version not restored: %d vs %d", s2.Version(), s1.Version())
	}

	// last consistent result is restored as well
	r2 := s2.Query()
	if r2 == nil {
		t.Fatal("last result not restored")
	}
	for id, m1 := range r1.Nodes {
		m2 := r2.Nodes[id]
		if math.Abs(m1.Pose.X-m2.Pose.X) > 1e-9 ||
			math.Abs(m1.Pose.Y-m2.Pose.Y) > 1e-9 {
			t.Fatalf("pose mismatch on restart for %s: %+v vs %+v", id, m1.Pose, m2.Pose)
		}
	}

	// incremental updates continue after restart: add node E and edge D-E
	s2.Submit(Change{Kind: ChangeNode, Node: Node{ID: "E", Initial: Pose2{1, 1, 2}}})
	s2.Submit(Change{Kind: ChangeConstraint, Constraint: Constraint{
		ID: "e4", A: "D", B: "E", Measurement: Pose2{-1, 0, 0},
		Covariance: diagCov(0.01, 0.01, 0.01)}})
	r3 := s2.Query()
	if r3 == nil || r3.Diag.Status != StatusConverged {
		t.Fatalf("post-restart incremental solve failed: %+v", r3)
	}
	e := r3.Nodes["E"].Pose
	// D=(2,1,pi/2); local (-1,0) rotated by pi/2 lands at world (0,-1),
	// so E = (2,0,pi/2).
	if math.Abs(e.X-2) > 1e-4 || math.Abs(e.Y-0) > 1e-4 ||
		math.Abs(normalizeAngle(e.Theta-math.Pi/2)) > 1e-4 {
		t.Fatalf("post-restart estimate wrong: %+v", e)
	}
}

func TestRejectedLoopPersisted(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	s1 := buildPersistentChain(t, path)
	bad := Constraint{ID: "bad", A: "D", B: "A", Loop: true,
		Measurement: Pose2{-2, 10, 0}, Covariance: diagCov(0.01, 0.01, 0.01)}
	rep, err := s1.Submit(Change{Kind: ChangeConstraint, Constraint: bad})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Reports[len(rep.Reports)-1].Outcome != OutcomeRejected {
		t.Fatal("expected reject before restart")
	}
	s1.Close()

	cfg := DefaultConfig()
	cfg.SnapshotPath = path
	s2, err := NewStore(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	if len(s2.Snapshot().Rejected) != 1 {
		t.Fatal("rejected loop not persisted across restart")
	}
	if len(s2.Snapshot().Constraints) != 3 {
		t.Fatal("rejected loop must not appear among accepted constraints")
	}
}

func TestSnapshotDeterministicJSON(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "A", Initial: Pose2{}, Fixed: true}})
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "B", Initial: Pose2{1, 0, 0}}})
	s.Submit(Change{Kind: ChangeConstraint, Constraint: Constraint{
		ID: "ab", A: "A", B: "B", Measurement: Pose2{1, 0, 0},
		Covariance: diagCov(0.01, 0.01, 0.01)}})
	b1, err := encodeSnapshot(s.Snapshot())
	if err != nil {
		t.Fatal(err)
	}
	b2, err := encodeSnapshot(s.Snapshot())
	if err != nil {
		t.Fatal(err)
	}
	if string(stripTimes(b1)) != string(stripTimes(b2)) {
		t.Fatal("snapshot serialization not stable apart from timestamps")
	}
}

// stripTimes removes wall-clock fields so map-key ordering determinism can be
// compared independently of timestamps.
func stripTimes(raw []byte) []byte {
	var v any
	if err := jsonUnmarshal(raw, &v); err != nil {
		return raw
	}
	var walk func(x any)
	walk = func(x any) {
		switch z := x.(type) {
		case map[string]any:
			delete(z, "saved_at")
			delete(z, "estimated_at")
			delete(z, "optimized_at")
			for _, v2 := range z {
				walk(v2)
			}
		case []any:
			for _, v2 := range z {
				walk(v2)
			}
		}
	}
	walk(v)
	out, err := encodeAny(v)
	if err != nil {
		return raw
	}
	return out
}
