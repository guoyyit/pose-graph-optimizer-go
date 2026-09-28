package posegraph

import "sort"

// initialGuess builds a deterministic initial pose for every node that is
// independent of constraint submission order:
//
//  1. The anchor (explicit or lexicographically smallest ID) is placed at
//     its hint (or identity).
//  2. Odometry/accepted edges form an undirected weighted graph; a BFS by
//     edge count (ties broken by edge ID, then neighbor ID) from the anchor
//     composes poses along the spanning tree.
//  3. Nodes unreachable from the anchor keep their hints; if also unhinted
//     they remain at identity and are flagged underconstrained by the caller.
func initialGuess(g *Graph, ids []string, index map[string]int, anchor int) []Pose {
	poses := make([]Pose, len(ids))
	for _, id := range ids {
		nd := g.nodes[id]
		poses[index[id]] = nd.Hint
	}
	anchorID := ids[anchor]
	poses[anchor] = g.nodes[anchorID].Hint

	// Adjacency from active edges, deterministic order:
	// smaller edge count first -> simple BFS queue; neighbor iteration order
	// is sorted by edge ID.
	adj := map[string][]neighbor{}
	for _, e := range g.activeEdges() {
		// Only odometry links define the spanning-tree initial guess; loop
		// closures are deliberately ignored here so their arbitrary order of
		// arrival cannot distort the linearization point.
		if e.Kind != KindOdometry {
			continue
		}
		adj[e.From] = append(adj[e.From], neighbor{to: e.To, edgeID: e.ID, z: e.Measurement, forward: true})
		adj[e.To] = append(adj[e.To], neighbor{to: e.From, edgeID: e.ID, z: e.Measurement, forward: false})
	}
	for k := range adj {
		sort.Slice(adj[k], func(i, j int) bool {
			if adj[k][i].edgeID != adj[k][j].edgeID {
				return adj[k][i].edgeID < adj[k][j].edgeID
			}
			return adj[k][i].to < adj[k][j].to
		})
	}

	visited := map[string]bool{anchorID: true}
	queue := []string{anchorID}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		tc := poses[index[cur]]
		for _, nb := range adj[cur] {
			if visited[nb.to] {
				continue
			}
			var rel Pose
			if nb.forward {
				rel = nb.z // T_cur^-1 * T_to
			} else {
				rel = Inverse(nb.z)
			}
			poses[index[nb.to]] = Compose(tc, rel)
			visited[nb.to] = true
			queue = append(queue, nb.to)
		}
	}
	return poses
}

type neighbor struct {
	to      string
	edgeID  string
	z       Pose
	forward bool
}
