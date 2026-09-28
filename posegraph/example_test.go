package posegraph_test

import (
	"fmt"

	"github.com/guoyyit/pose-graph-optimizer-go/posegraph"
)

func ExampleNewStore() {
	store, _ := posegraph.NewStore(posegraph.DefaultConfig())

	store.Submit(posegraph.Change{Kind: posegraph.ChangeNode,
		Node: posegraph.Node{ID: "A", Initial: posegraph.Pose2{}, Fixed: true}})
	store.Submit(posegraph.Change{Kind: posegraph.ChangeNode,
		Node: posegraph.Node{ID: "B", Initial: posegraph.Pose2{X: 1.2, Y: 0.3}}})

	store.Submit(posegraph.Change{Kind: posegraph.ChangeConstraint,
		Constraint: posegraph.Constraint{
			ID:          "ab",
			A:           "A",
			B:           "B",
			Measurement: posegraph.Pose2{X: 1},
			Covariance:  posegraph.Mat3{0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01},
		}})

	res := store.Query()
	fmt.Println(res.Diag.Status == posegraph.StatusConverged)
	fmt.Printf("B=(%.2f,%.2f)\n", res.Nodes["B"].Pose.X, res.Nodes["B"].Pose.Y)
	// Output:
	// true
	// B=(1.00,0.00)
}
