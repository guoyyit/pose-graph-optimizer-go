package posegraph

import (
	"fmt"
	"math"
	"time"
)

// optimizeOutput bundles the result of a solver run.
type optimizeOutput struct {
	poses       map[NodeID]Pose2
	marginals   map[NodeID]Marginal
	residuals   map[string]Residual
	perNode     map[NodeID]NodeResidual
	diag        Diagnostics
	loopWeights map[string]float64
	ok          bool
}

// block is one endpoint participation in a linearized constraint.
type block struct {
	id  NodeID
	idx int
	j   Mat3
}

// constraintInfo resolves the whitening information for a constraint, applying
// the configured diagonal floor to the covariance. An explicitly supplied
// Information matrix takes precedence.
func constraintInfo(c Constraint, floor Vec3) Mat3 {
	if c.Information != (Mat3{}) {
		return symmetrize3(c.Information)
	}
	info, ok := covarianceInformation(c.Covariance, floor)
	if !ok {
		return matScale(matIdentity(), 1e12)
	}
	return info
}

// gaugeFixed returns the set of nodes held fixed for gauge/anchoring reasons:
// explicitly fixed nodes, plus (only when no prior exists) the
// lexicographically smallest free node chosen deterministically.
func gaugeFixed(g *Graph) (map[NodeID]bool, NodeID) {
	fixed := map[NodeID]bool{}
	for _, id := range g.sortedNodeIDs() {
		if n, _ := g.getNode(id); n.Fixed {
			fixed[id] = true
		}
	}
	anchor := NodeID("")
	if len(fixed) == 0 && !hasPrior(g) {
		anchor = anchorNode(g, fixed)
		if anchor != "" {
			fixed[anchor] = true
		}
	}
	return fixed, anchor
}

// anchorNode deterministically chooses the gauge anchor: lexicographically
// smallest free node. The choice depends solely on the ID set, not on order.
func anchorNode(g *Graph, fixed map[NodeID]bool) NodeID {
	var anchor NodeID
	for _, id := range g.sortedNodeIDs() {
		if fixed[id] {
			continue
		}
		if anchor == "" || id < anchor {
			anchor = id
		}
	}
	return anchor
}

// hasPrior reports whether the graph contains an accepted prior.
func hasPrior(g *Graph) bool {
	for _, id := range g.sortedConstraintIDs() {
		if c, ok := g.getConstraint(id); ok && c.Kind == Prior {
			return true
		}
	}
	return false
}

// runOptimization solves the pose graph. warmStart (nil) initializes from node
// Initial poses. It never returns a high-error solution flagged as converged:
// the status distinguishes convergence, iteration cap, stagnation and numeric
// failure.
func runOptimization(cfg Config, g *Graph, warmStart map[NodeID]Pose2) optimizeOutput {
	ids := g.sortedNodeIDs()
	idx := make(map[NodeID]int, len(ids))
	poses := make(map[NodeID]Pose2, len(ids))
	for i, id := range ids {
		idx[id] = i
		if p, ok := warmStart[id]; ok && warmStart != nil {
			poses[id] = p
			continue
		}
		n, _ := g.getNode(id)
		poses[id] = Pose2{n.Initial.X, n.Initial.Y, normalizeAngle(n.Initial.Theta)}
	}

	fixed, _ := gaugeFixed(g)
	n := len(ids)
	diag := Diagnostics{Status: StatusNotRun, NumNodes: n, NumConstraints: g.constraintCount(),
		OptimizedAt: time.Now()}
	out := optimizeOutput{poses: poses, diag: diag, loopWeights: map[string]float64{}}

	if n == 0 {
		out.diag.Status = StatusFailed
		out.diag.Warnings = []string{"empty graph"}
		return out
	}
	if !graphConnected(g, fixed) {
		out.diag.Status = StatusIllConditioned
		out.diag.Warnings = []string{"unconstrained singular component after fixing nodes"}
		out.marginals = map[NodeID]Marginal{}
		return out
	}

	weights := make(map[string]float64)
	for _, id := range g.sortedConstraintIDs() {
		if c, _ := g.getConstraint(id); c.RobustWeight > 0 {
			weights[id] = c.RobustWeight
		} else {
			weights[id] = 1
		}
	}

	eval := func(now map[NodeID]Pose2) (h []float64, b []float64, cost float64,
		res map[string]Residual, pn map[NodeID]NodeResidual) {
		dim := n * 3
		h = make([]float64, dim*dim)
		b = make([]float64, dim)
		res, pn = evaluateResidualsWith(g, now, weights, cfg)
		for _, cid := range g.sortedConstraintIDs() {
			c, _ := g.getConstraint(cid)
			w := weights[cid]
			if w <= 0 {
				continue
			}
			winfo := matScale(constraintInfo(c, cfg.CovarianceFloor), w)
			blocks := linearBlocks(c, now, idx)
			e := constraintError(c, now)
			accumulate(h, b, winfo, blocks, e, fixed, n)
			cost += eDot(winfo, e)
		}
		return h, b, cost, res, pn
	}

	_, _, cost, res, pn := eval(poses)
	initialCost := cost
	damping := cfg.InitialDamping
	stagnant := 0
	var lastStep, cond float64
	status := StatusMaxIterations

	// Already at a (numerically) zero-residual configuration.
	if cost <= cfg.ConvergeChi2 {
		status = StatusConverged
		diag.Iterations = 0
	}

	for it := 0; status != StatusConverged && it < cfg.MaxIterations; it++ {
		h, bvec, _, _, _ := eval(poses)
		free := freeMapping(n, ids, fixed)
		hf, bf := freeSystem(h, bvec, free, n)
		// Damping is applied to the free block only, scaled by its diagonal.
		for i := range bf {
			hf[i*len(bf)+i] += damping * math.Max(hf[i*len(bf)+i], 1e-12)
		}
		l, err := cholesky(hf, len(bf))
		if err != nil {
			status = StatusIllConditioned
			diag.Warnings = append(diag.Warnings, "normal-equations factorization failed")
			break
		}
		rc := reciprocalCondition(l, len(bf))
		if rc > 0 {
			cond = 1 / rc // report the actual condition-number estimate
		} else {
			cond = math.Inf(1)
		}
		if rc > 0 && 1/rc > cfg.ConditionLimit {
			status = StatusIllConditioned
			diag.Warnings = append(diag.Warnings, "matrix exceeds condition-number limit")
			break
		}
		step := solveStep(l, free, bvec, n)
		lastStep = normVec(step)
		cand := applyStep(poses, ids, step)
		if !ensureFinitePoses(cand) {
			status = StatusFailed
			diag.Warnings = append(diag.Warnings, "non-finite step")
			break
		}
		_, _, newCost, newRes, newPN := eval(cand)
		if newCost < cost {
			delta := cost - newCost
			poses, cost, res, pn = cand, newCost, newRes, newPN
			damping *= cfg.DampingDown
			stagnant = 0
			updateWeights(g, weights, res, cfg, out.loopWeights)
			if newCost <= cfg.ConvergeChi2 ||
				delta <= cfg.ConvergeChi2*math.Max(1, initialCost) ||
				lastStep < cfg.ConvergeStep {
				status = StatusConverged
				diag.Iterations = it + 1
				break
			}
		} else {
			damping *= cfg.DampingUp
			stagnant++
			if damping > cfg.MaxDamping {
				status = StatusFailed
				diag.Warnings = append(diag.Warnings, "damping exceeded limit; no descent")
				break
			}
			if stagnant >= cfg.StagnationRounds {
				// Stopped progressing; call it convergence only when residuals
				// are already negligibly small, otherwise genuine stagnation.
				if cost <= cfg.ConvergeChi2 {
					status = StatusConverged
				} else {
					status = StatusStagnated
					diag.Warnings = append(diag.Warnings, "optimizer stopped making progress")
				}
				break
			}
		}
		diag.Iterations = it + 1
	}

	// Final consistent evaluation at reported poses with final weights.
	updateWeights(g, weights, res, cfg, out.loopWeights)
	_, _, finalCost, res, pn := eval(poses)
	sum2, maxW, rmse := chi2Stats(res)

	diag.Status = status
	diag.FinalChi2 = finalCost
	diag.MaxWhitened = maxW
	diag.RMSE = rmse
	diag.StepNorm = lastStep
	diag.ConditionEst = cond

	out.poses = poses
	out.residuals = res
	out.perNode = pn
	out.diag = diag
	// A run is acceptable as a new published version only when:
	//  - the solve genuinely converged (or hit the iteration cap while still
	//    descending), AND
	//  - no individual constraint is grossly violated. A converged least-squares
	//    compromise between mutually contradictory hard constraints is reported
	//    as StatusConverged but is NOT a success-quality result; its diagnostics
	//    remain available via Optimize()/CommitReport.
	highResidual := maxW > highResidualGate(cfg)
	out.ok = (status == StatusConverged || status == StatusMaxIterations) && !highResidual
	if status == StatusStagnated || status == StatusIllConditioned || status == StatusFailed {
		out.ok = false
	}
	if highResidual && status == StatusConverged {
		out.diag.Status = StatusStagnated
		out.diag.Warnings = append(out.diag.Warnings,
			fmt.Sprintf("converged to a high-residual compromise (max whitened %.2f); not marked successful", maxW))
		out.ok = false
	}
	out.marginals = computeMarginals(g, poses, weights, cfg)
	_ = sum2
	return out
}

// highResidualGate is the per-constraint whitened norm above which even a
// converged solution contains an unresolved inconsistency and must not be
// marked successful. It corresponds to a conservative chi-square tail (~sqrt
// of 25, roughly 5-sigma on one component) and is intentionally independent of
// the loop-closure admission gate.
func highResidualGate(_ Config) float64 {
	return 5
}

// linearBlocks builds the endpoint/Jacobian blocks for one constraint.
func linearBlocks(c Constraint, now map[NodeID]Pose2, idx map[NodeID]int) []block {
	if c.Kind == Prior {
		return []block{{c.B, idx[c.B], priorJacobianWorld(c.Measurement)}}
	}
	ja, jb := relJacobiansWorld(now[c.A], now[c.B], c.Measurement)
	return []block{
		{c.A, idx[c.A], ja},
		{c.B, idx[c.B], jb},
	}
}

// constraintError returns the raw (unwhitened) error of one constraint.
func constraintError(c Constraint, now map[NodeID]Pose2) Vec3 {
	if c.Kind == Prior {
		return relativeError(c.Measurement, now[c.B])
	}
	return relativeError(c.Measurement, between(now[c.A], now[c.B]))
}

// graphConnected reports whether every free node is reachable via edges from a
// fixed/prior-rooted node, ruling out structurally singular components.
func graphConnected(g *Graph, fixed map[NodeID]bool) bool {
	adj := map[NodeID]map[NodeID]bool{}
	var stack []NodeID
	for _, id := range g.sortedNodeIDs() {
		adj[id] = map[NodeID]bool{}
		if fixed[id] {
			stack = append(stack, id)
		}
	}
	for _, cid := range g.sortedConstraintIDs() {
		c, _ := g.getConstraint(cid)
		if c.Kind == Prior {
			stack = append(stack, c.B)
			continue
		}
		adj[c.A][c.B] = true
		adj[c.B][c.A] = true
	}
	seen := map[NodeID]bool{}
	for len(stack) > 0 {
		x := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[x] {
			continue
		}
		seen[x] = true
		for y := range adj[x] {
			if !seen[y] {
				stack = append(stack, y)
			}
		}
	}
	for _, id := range g.sortedNodeIDs() {
		if !fixed[id] && !seen[id] {
			return false
		}
	}
	return true
}

// accumulate adds one constraint's Gauss-Newton contribution H += Jᵀ W J and
// b -= Jᵀ W e, skipping fixed endpoint blocks entirely.
func accumulate(h []float64, b []float64, w Mat3, blocks []block, e Vec3, fixed map[NodeID]bool, n int) {
	dim := n * 3
	we := matVec3(w, e)
	for _, bx := range blocks {
		if fixed[bx.id] {
			continue
		}
		jte := matVec3(transpose3(bx.j), we)
		for r := 0; r < 3; r++ {
			b[bx.idx*3+r] -= jte[r]
		}
		for _, by := range blocks {
			if fixed[by.id] {
				continue
			}
			jtw := matMul3(transpose3(bx.j), matMul3(w, by.j))
			baseR, baseC := bx.idx*3, by.idx*3
			for r := 0; r < 3; r++ {
				for c := 0; c < 3; c++ {
					h[(baseR+r)*dim+(baseC+c)] += jtw[r*3+c]
				}
			}
		}
	}
}

func eDot(w Mat3, e Vec3) float64 {
	we := matVec3(w, e)
	return e[0]*we[0] + e[1]*we[1] + e[2]*we[2]
}

// freeMapping maps node index -> free-vector position (-1 when fixed).
func freeMapping(n int, ids []NodeID, fixed map[NodeID]bool) []int {
	mp := make([]int, n)
	pos := 0
	for i, id := range ids {
		if fixed[id] {
			mp[i] = -1
		} else {
			mp[i] = pos
			pos++
		}
	}
	return mp
}

func freeSystem(h, b []float64, free []int, n int) ([]float64, []float64) {
	dim := 0
	for _, p := range free {
		if p >= 0 {
			dim++
		}
	}
	nf := dim * 3
	hf := make([]float64, nf*nf)
	bf := make([]float64, nf)
	for i, pi := range free {
		if pi < 0 {
			continue
		}
		copy(bf[pi*3:pi*3+3], b[i*3:i*3+3])
		for j, pj := range free {
			if pj < 0 {
				continue
			}
			for r := 0; r < 3; r++ {
				copy(hf[(pi*3+r)*nf+pj*3:(pi*3+r)*nf+pj*3+3],
					h[(i*3+r)*n*3+j*3:(i*3+r)*n*3+j*3+3])
			}
		}
	}
	return hf, bf
}

func solveStep(l []float64, free []int, b []float64, n int) []float64 {
	dim := 0
	for _, p := range free {
		if p >= 0 {
			dim++
		}
	}
	nf := dim * 3
	bf := make([]float64, nf)
	for i, p := range free {
		if p >= 0 {
			copy(bf[p*3:p*3+3], b[i*3:i*3+3])
		}
	}
	xf := solveCholesky(l, nf, bf)
	step := make([]float64, n*3)
	for i, p := range free {
		if p >= 0 {
			copy(step[i*3:i*3+3], xf[p*3:p*3+3])
		}
	}
	return step
}

func applyStep(base map[NodeID]Pose2, ids []NodeID, step []float64) map[NodeID]Pose2 {
	out := make(map[NodeID]Pose2, len(ids))
	for i, id := range ids {
		p := base[id]
		p.X += step[i*3+0]
		p.Y += step[i*3+1]
		p.Theta = normalizeAngle(p.Theta + step[i*3+2])
		out[id] = p
	}
	return out
}

func normVec(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

// ensureFinitePoses rejects non-finite estimates.
func ensureFinitePoses(poses map[NodeID]Pose2) bool {
	for _, p := range poses {
		if !poseFinite(p) {
			return false
		}
	}
	return true
}

// evaluateResidualsWith computes whitened residuals and per-node aggregates
// using supplied robust weights.
func evaluateResidualsWith(g *Graph, poses map[NodeID]Pose2, weights map[string]float64, cfg Config) (map[string]Residual, map[NodeID]NodeResidual) {
	res := map[string]Residual{}
	pn := map[NodeID]NodeResidual{}
	for _, id := range g.sortedNodeIDs() {
		pn[id] = NodeResidual{NodeID: id}
	}
	for _, cid := range g.sortedConstraintIDs() {
		c, _ := g.getConstraint(cid)
		w := math.Max(weights[cid], 0)
		info := matScale(constraintInfo(c, cfg.CovarianceFloor), w)
		sqrtInfo, ok := cholSqrt3(info)
		if !ok {
			sqrtInfo = matIdentity()
		}
		e := constraintError(c, poses)
		wh := matVec3(sqrtInfo, e)
		n2 := math.Sqrt(wh[0]*wh[0] + wh[1]*wh[1] + wh[2]*wh[2])
		res[cid] = Residual{ConstraintID: cid, Raw: e, Whitened: wh, Norm2: n2}
		we := matVec3(info, e)
		touch := func(nid NodeID) {
			v := pn[nid]
			v.Force[0] += we[0]
			v.Force[1] += we[1]
			v.Force[2] += we[2]
			v.Degree++
			if n2 > v.MaxWhitened {
				v.MaxWhitened = n2
			}
			pn[nid] = v
		}
		if c.Kind == Prior {
			touch(c.B)
		} else {
			touch(c.A)
			touch(c.B)
		}
	}
	return res, pn
}

// evaluateResiduals is the unit-weight convenience wrapper.
func evaluateResiduals(g *Graph, poses map[NodeID]Pose2) (map[string]Residual, map[NodeID]NodeResidual) {
	w := map[string]float64{}
	for _, cid := range g.sortedConstraintIDs() {
		w[cid] = 1
	}
	return evaluateResidualsWith(g, poses, w, DefaultConfig())
}

// updateWeights performs one IRLS sweep: loop constraints whose whitened
// residual exceeds the kernel delta get a DCS-style weight; others stay at 1.
func updateWeights(g *Graph, weights map[string]float64, res map[string]Residual, cfg Config, loopWeights map[string]float64) {
	for _, cid := range g.sortedConstraintIDs() {
		c, _ := g.getConstraint(cid)
		if !c.Loop {
			weights[cid] = 1
			continue
		}
		wt := robustWeight(cfg, res[cid].Norm2)
		weights[cid] = wt
		loopWeights[cid] = wt
	}
}

// computeMarginals inverts the free block of the converged information matrix.
// Fixed/anchored nodes report zero covariance.
func computeMarginals(g *Graph, poses map[NodeID]Pose2, weights map[string]float64, cfg Config) map[NodeID]Marginal {
	ids := g.sortedNodeIDs()
	idx := make(map[NodeID]int, len(ids))
	for i, id := range ids {
		idx[id] = i
	}
	fixed, _ := gaugeFixed(g)
	n := len(ids)
	dim := n * 3
	h := make([]float64, dim*dim)
	for _, cid := range g.sortedConstraintIDs() {
		c, _ := g.getConstraint(cid)
		w := weights[cid]
		if w <= 0 {
			continue
		}
		winfo := matScale(constraintInfo(c, cfg.CovarianceFloor), w)
		blocks := linearBlocks(c, poses, idx)
		e := constraintError(c, poses)
		accumulateMarginal(h, winfo, blocks, e, fixed, n)
	}
	free := freeMapping(n, ids, fixed)
	hf, _ := freeSystem(h, make([]float64, dim), free, n)
	nf := 0
	for _, p := range free {
		if p >= 0 {
			nf++
		}
	}
	out := make(map[NodeID]Marginal, n)
	for _, id := range ids {
		out[id] = Marginal{ID: id, Pose: poses[id], Fixed: fixed[id]}
	}
	l, err := cholesky(hf, nf*3)
	if err != nil || nf == 0 {
		return out
	}
	cov := inverseCholesky(l, nf*3)
	for i, p := range free {
		if p < 0 {
			continue
		}
		var blk Mat3
		for r := 0; r < 3; r++ {
			for c := 0; c < 3; c++ {
				blk[r*3+c] = cov[(p*3+r)*nf*3+(p*3+c)]
			}
		}
		m := out[ids[i]]
		m.Covariance = symmetrize3(blk)
		out[ids[i]] = m
	}
	return out
}

func accumulateMarginal(h []float64, w Mat3, blocks []block, e Vec3, fixed map[NodeID]bool, n int) {
	dim := n * 3
	for _, bx := range blocks {
		if fixed[bx.id] {
			continue
		}
		for _, by := range blocks {
			if fixed[by.id] {
				continue
			}
			jtw := matMul3(transpose3(bx.j), matMul3(w, by.j))
			baseR, baseC := bx.idx*3, by.idx*3
			for r := 0; r < 3; r++ {
				for c := 0; c < 3; c++ {
					h[(baseR+r)*dim+(baseC+c)] += jtw[r*3+c]
				}
			}
		}
	}
}

// chi2Stats summarizes residual cost, max whitened norm and component RMSE.
func chi2Stats(res map[string]Residual) (sum2, maxW, rmse float64) {
	components := 0
	for _, r := range res {
		sum2 += r.Whitened[0]*r.Whitened[0] + r.Whitened[1]*r.Whitened[1] + r.Whitened[2]*r.Whitened[2]
		if r.Norm2 > maxW {
			maxW = r.Norm2
		}
		components += 3
	}
	if components > 0 {
		rmse = math.Sqrt(sum2 / float64(components))
	}
	return sum2, maxW, rmse
}
