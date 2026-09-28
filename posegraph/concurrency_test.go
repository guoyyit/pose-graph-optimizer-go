package posegraph

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
)

func TestConcurrentCommitQuery(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	// seed anchor
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "n0", Initial: Pose2{}, Fixed: true}})

	var writers, readers sync.WaitGroup
	stop := make(chan struct{})

	// writers: each owns a unique node and links it to the anchor
	for w := 0; w < 8; w++ {
		writers.Add(1)
		go func(w int) {
			defer writers.Done()
			id := NodeID(fmt.Sprintf("leaf-%d", w))
			n := Node{ID: id, Initial: Pose2{float64(w + 1), 0, 0}}
			if _, err := s.Submit(Change{Kind: ChangeNode, Node: n}); err != nil {
				t.Errorf("node submit: %v", err)
				return
			}
			c := Constraint{
				ID: fmt.Sprintf("edge-%d", w), A: "n0", B: id,
				Measurement: Pose2{float64(w + 1), 0, 0},
				Covariance:  diagCov(0.05, 0.05, 0.05),
			}
			if _, err := s.Submit(Change{Kind: ChangeConstraint, Constraint: c}); err != nil {
				t.Errorf("constraint submit: %v", err)
			}
		}(w)
	}

	// readers: every observed result must be internally consistent — same
	// version across maps, finite poses, no partially applied commit.
	for r := 0; r < 4; r++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				res := s.Query()
				if res != nil {
					if res.Version < 0 {
						t.Errorf("bad version %d", res.Version)
					}
					for id, m := range res.Nodes {
						if id == "" || !poseFinite(m.Pose) {
							t.Errorf("inconsistent node snapshot: %+v", m)
						}
					}
					if res.Diag.NumNodes != len(res.Nodes) {
						t.Errorf("torn read: diag nodes %d vs map %d",
							res.Diag.NumNodes, len(res.Nodes))
					}
				}
				runtime.Gosched()
			}
		}()
	}

	writers.Wait()
	close(stop)
	readers.Wait()

	final := s.Query()
	if final == nil {
		t.Fatal("no final result")
	}
	if final.Diag.NumNodes != 9 {
		t.Fatalf("want 9 nodes, got %d", final.Diag.NumNodes)
	}
	// mutating a returned result must not affect the store (deep copy)
	final.Nodes["n0"] = Marginal{}
	again := s.Query()
	if again.Nodes["n0"].ID != "n0" {
		t.Fatal("Query result was not isolated (deep copy missing)")
	}
}

func TestWaitForVersion(t *testing.T) {
	s, _ := NewStore(DefaultConfig())
	got := make(chan *Result, 1)
	go func() {
		r, ok := s.WaitForVersion(1)
		if !ok {
			got <- nil
			return
		}
		got <- r
	}()
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "A", Initial: Pose2{}, Fixed: true}})
	s.Submit(Change{Kind: ChangeNode, Node: Node{ID: "B", Initial: Pose2{1, 0, 0}}})
	s.Submit(Change{Kind: ChangeConstraint, Constraint: Constraint{
		ID: "ab", A: "A", B: "B", Measurement: Pose2{1, 0, 0},
		Covariance: diagCov(0.01, 0.01, 0.01)}})
	r := <-got
	if r == nil || r.Version < 1 {
		t.Fatal("waiter did not observe version 1")
	}

	// closing unblocks waiters
	s.Close()
	if _, ok := s.WaitForVersion(1 << 40); ok {
		t.Fatal("waiter should unblock with false on close")
	}
}
