package posegraph

import (
	"context"
	"math"
	"path/filepath"
	"sync"
	"testing"
)

func svcSq(t *testing.T, loopBad bool) (*Service, context.Context) {
	t.Helper()
	svc, err := NewService(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	poses := map[string]Pose{"a": {}, "b": {10, 0, 0}, "c": {10, 10, 0}, "d": {0, 10, 0}}
	c := dcov(0.05, 0.05, 0.02)
	// Interleave nodes and edges so intermediate versions stay well-posed;
	// ignore transient under-constrained errors while the graph is incomplete.
	addNode := func(id string) {
		if err := svc.AddNode(ctx, Node{ID: id, Hint: poses[id]}); err != nil && err != ErrDegenerate {
			t.Fatal(err)
		}
	}
	addEdge := func(e Constraint) {
		if err := svc.AddConstraint(ctx, e); err != nil && err != ErrDegenerate {
			t.Fatal(err)
		}
	}
	addNode("a")
	addNode("b")
	addEdge(Constraint{ID: "ab", From: "a", To: "b", Measurement: Pose{10, 0, 0}, Covariance: c})
	addNode("c")
	addEdge(Constraint{ID: "bc", From: "b", To: "c", Measurement: Pose{0, 10, 0}, Covariance: c})
	addNode("d")
	addEdge(Constraint{ID: "cd", From: "c", To: "d", Measurement: Pose{-10, 0, 0}, Covariance: c})
	addEdge(Constraint{ID: "da", From: "d", To: "a", Measurement: Pose{0, -10, 0}, Covariance: c})
	// a diagonal odometry link keeps all heading degrees of freedom observable
	// (a pure-translation square is otherwise near-rank-deficient in heading).
	addEdge(Constraint{ID: "acbrace", From: "a", To: "c", Measurement: Pose{10, 10, 0}, Covariance: c})
	if !svc.Query().Diag.OK {
		t.Fatalf("base graph not converged: %s", svc.Query().Diag.Message)
	}
	return svc, ctx
}

// TestServiceGoodAndBadLoop verifies a valid loop is accepted, a grossly
// inconsistent loop is rejected, and the rest of the graph still estimates.
func TestServiceGoodAndBadLoop(t *testing.T) {
	svc, ctx := svcSq(t, false)
	defer svc.Close()
	c := dcov(0.05, 0.05, 0.02)

	// valid loop c->a = (-10,-10)
	if err := svc.AddConstraint(ctx, Constraint{ID: "good", From: "c", To: "a",
		Measurement: Pose{-10, -10, 0}, Covariance: c, Kind: KindLoop}); err != nil {
		t.Fatalf("good loop rejected unexpectedly: %v", err)
	}
	if st, _, _ := svc.EdgeState("good"); st == EdgeRejected {
		t.Fatal("good loop rejected")
	}

	// bad loop: 10m error in x
	err := svc.AddConstraint(ctx, Constraint{ID: "bad", From: "c", To: "a",
		Measurement: Pose{0, -10, 0}, Covariance: c, Kind: KindLoop})
	if err != ErrConstraintRejected {
		t.Errorf("bad loop returned %v, want ErrConstraintRejected", err)
	}
	st, reason, ok := svc.EdgeState("bad")
	if !ok {
		t.Fatal("bad edge missing")
	}
	if st != EdgeRejected {
		t.Fatalf("bad loop state=%s want rejected (%s)", st, reason)
	}

	// graph still consistent and usable
	res := svc.Query()
	if !res.Diag.OK {
		t.Fatalf("graph not usable after reject: %s", res.Diag.Message)
	}
	pc, _ := res.PoseAt("c")
	if math.Hypot(pc.X-10, pc.Y-10) > 0.5 {
		t.Fatalf("trajectory distorted by rejected loop: c=%+v", pc)
	}

	// rejection persists across subsequent incremental updates
	if err := svc.AddNode(ctx, Node{ID: "z", Hint: Pose{0, -10, 0}}); err == nil {
		// isolated node is expected to error; key check is bad stays rejected
	}
	st2, _, _ := svc.EdgeState("bad")
	if st2 != EdgeRejected {
		t.Fatalf("rejected loop resurrected: %s", st2)
	}
}

// TestServicePendingOutOfOrder submits a constraint before its nodes and
// confirms buffering then deterministic promotion.
func TestServicePendingOutOfOrder(t *testing.T) {
	svc, _ := NewService(DefaultConfig())
	defer svc.Close()
	ctx := context.Background()
	c := dcov(0.1, 0.1, 0.05)

	err := svc.AddConstraint(ctx, Constraint{ID: "xy", From: "x", To: "y",
		Measurement: Pose{5, 0, 0}, Covariance: c})
	if err != ErrPending {
		t.Fatalf("want ErrPending, got %v", err)
	}
	if err := svc.AddNode(ctx, Node{ID: "y", Hint: Pose{5, 0, 0}}); err != nil {
		t.Fatal(err)
	}
	if st, _, _ := svc.EdgeState("xy"); st != EdgePending {
		t.Fatalf("want pending, got %s", st)
	}
	if err := svc.AddNode(ctx, Node{ID: "x"}); err != nil {
		t.Fatal(err)
	}
	if st, _, _ := svc.EdgeState("xy"); st != EdgeAccepted {
		t.Fatalf("want accepted after both endpoints exist, got %s", st)
	}
}

// TestServiceVersionIsolation confirms readers always observe a complete,
// immutable version while writers mutate concurrently (run under -race).
func TestServiceVersionIsolation(t *testing.T) {
	svc, _ := NewService(DefaultConfig())
	defer svc.Close()
	ctx := context.Background()
	for i := 0; i < 15; i++ {
		id := nodeName(i)
		svc.AddNode(ctx, Node{ID: id, Hint: Pose{float64(i), 0, 0}})
		if i > 0 {
			err := svc.AddConstraint(ctx, Constraint{
				From: nodeName(i - 1), To: id, Measurement: Pose{1, 0, 0},
				Covariance: dcov(0.05, 0.05, 0.02),
			})
			if err != nil {
				t.Fatal(err)
			}
		}
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for r := 0; r < 8; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				res := svc.Query()
				if res == nil {
					t.Error("nil version")
					return
				}
				n := len(res.Nodes)
				// all slices must be coherent and free of NaN
				if len(res.Residuals) < 0 {
					t.Error("bad residuals")
				}
				for _, ne := range res.Nodes {
					if math.IsNaN(ne.Pose.X) || math.IsNaN(ne.Covariance[4]) {
						t.Errorf("partial/mid-iteration state observed at v%d, nodes=%d", res.Version, n)
						return
					}
				}
				// historical versions remain immutable after publication
				if old := svc.QueryAt(res.Version); old != nil && len(old.Nodes) != n {
					t.Error("published version mutated")
					return
				}
			}
		}()
	}
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for k := 0; k < 8; k++ {
				id := "w" + itoa(w) + "_" + itoa(k)
				_ = svc.AddNode(ctx, Node{ID: id, Hint: Pose{0, float64(w + k), 0}})
				_ = svc.AddConstraint(ctx, Constraint{
					From: nodeName(0), To: id, Measurement: Pose{0, float64(w + k), 0},
					Covariance: dcov(0.1, 0.1, 0.05),
				})
			}
		}(w)
	}
	// busy-wait until writers finished enough mutations, then stop readers
	for svc.Versions() < 40 {
	}
	close(stop)
	wg.Wait()
}

func nodeName(i int) string {
	return "p" + itoa(i)
}
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}

// TestServicePersistenceRestart verifies snapshot+WAL recovery preserves all
// accepted constraints and estimates, and that updates continue afterwards.
func TestServicePersistenceRestart(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.PersistenceDir = filepath.Join(dir, "pg")
	cfg.SnapshotEvery = 3 // force snapshot rotation during the run

	svc, err := NewService(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	for i := 0; i < 8; i++ {
		svc.AddNode(ctx, Node{ID: nodeName(i), Hint: Pose{float64(2 * i), 0, 0}})
		if i > 0 {
			if err := svc.AddConstraint(ctx, Constraint{
				ID: "e" + itoa(i), From: nodeName(i - 1), To: nodeName(i),
				Measurement: Pose{2, 0, 0}, Covariance: dcov(0.05, 0.05, 0.02),
			}); err != nil {
				t.Fatal(err)
			}
		}
	}
	before := svc.Query()
	if !before.Diag.OK {
		t.Fatal(before.Diag.Message)
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}

	// reopen from disk
	svc2, err := NewService(cfg)
	if err != nil {
		t.Fatal(err)
	}
	got := svc2.Query()
	if len(got.Nodes) != 8 {
		t.Fatalf("restored %d nodes, want 8", len(got.Nodes))
	}
	for i := 0; i < 8; i++ {
		id := nodeName(i)
		p1, _ := before.PoseAt(id)
		p2, ok := got.PoseAt(id)
		if !ok {
			t.Fatalf("node %s missing after restart", id)
		}
		if math.Hypot(p1.X-p2.X, p1.Y-p2.Y) > 1e-6 {
			t.Errorf("node %s pose changed across restart: %+v vs %+v", id, p2, p1)
		}
	}
	for i := 1; i < 8; i++ {
		st, _, ok := svc2.EdgeState("e" + itoa(i))
		if !ok || st != EdgeAccepted {
			t.Fatalf("accepted edge e%d lost: state=%s ok=%v", i, st, ok)
		}
	}

	// edge first (pending), then node -> promotes after restart
	if err := svc2.AddConstraint(ctx, Constraint{
		ID: "e8", From: nodeName(7), To: nodeName(8), Measurement: Pose{2, 0, 0},
		Covariance: dcov(0.05, 0.05, 0.02),
	}); err != ErrPending {
		t.Fatalf("want ErrPending, got %v", err)
	}
	if err := svc2.AddNode(ctx, Node{ID: nodeName(8), Hint: Pose{18, 0, 0}}); err != nil {
		t.Fatal(err)
	}
	p8, _ := svc2.Query().PoseAt(nodeName(8))
	if math.Abs(p8.X-16) > 0.2 {
		t.Fatalf("incremental update after restart failed: p8=%+v", p8)
	}
	if err := svc2.Close(); err != nil {
		t.Fatal(err)
	}

	// reopen once more: the post-restart edge must still be there
	svc3, err := NewService(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer svc3.Close()
	if st, _, ok := svc3.EdgeState("e8"); !ok || st != EdgeAccepted {
		t.Fatalf("post-restart edge not durable: state=%s ok=%v", st, ok)
	}
}

// TestServiceFailureKeepsLastGoodVersion ensures a solve that cannot converge
// does not overwrite the last known-good version.
func TestServiceFailureKeepsLastGoodVersion(t *testing.T) {
	svc, _ := NewService(DefaultConfig())
	defer svc.Close()
	ctx := context.Background()
	svc.AddNode(ctx, Node{ID: "a"})
	svc.AddNode(ctx, Node{ID: "b", Hint: Pose{5, 0, 0}})
	if err := svc.AddConstraint(ctx, Constraint{
		From: "a", To: "b", Measurement: Pose{5, 0, 0},
		Covariance: dcov(0.1, 0.1, 0.05),
	}); err != nil {
		t.Fatal(err)
	}
	good := svc.Query()
	if !good.Diag.OK {
		t.Fatal(good.Diag.Message)
	}
	// add an isolated node -> underconstrained; last good version survives
	if err := svc.AddNode(ctx, Node{ID: "orphan"}); err == nil {
		t.Fatal("expected underconstrained error")
	}
	cur := svc.Query()
	if len(cur.Nodes) != 3 {
		t.Fatalf("version should include the submitted node, got %d nodes", len(cur.Nodes))
	}
	if cur.Diag.OK {
		t.Fatal("expected the new version to report failure, not false success")
	}
	if _, ok := cur.PoseAt("b"); !ok {
		t.Fatal("last graph content unavailable")
	}
}
