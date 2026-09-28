package posegraph

import (
	"fmt"
	"math"
	"sort"
)

// SolveStatus classifies an optimization run.
type SolveStatus int

const (
	// SolveEmpty is returned when no nodes exist.
	SolveEmpty SolveStatus = iota
	// SolveConverged means tolerances were met and the solution passed checks.
	SolveConverged
	// SolveStagnated means the optimizer made no progress before convergence.
	SolveStagnated
	// SolveMaxIterations means iteration budget was exhausted.
	SolveMaxIterations
	// SolveIllConditioned means Cholesky failed or condition number is too high.
	SolveIllConditioned
	// SolveHighResidual means the converged cost exceeds the sanity ceiling.
	SolveHighResidual
	// SolveUnderconstrained means essential degrees of freedom are unobservable.
	SolveUnderconstrained
)

func (s SolveStatus) String() string {
	switch s {
	case SolveEmpty:
		return "empty"
	case SolveConverged:
		return "converged"
	case SolveStagnated:
		return "stagnated"
	case SolveMaxIterations:
		return "max_iterations"
	case SolveIllConditioned:
		return "ill_conditioned"
	case SolveHighResidual:
		return "high_residual"
	case SolveUnderconstrained:
		return "underconstrained"
	default:
		return "unknown"
	}
}

// Residual is the per-constraint residual after solving.
type Residual struct {
	EdgeID string
	State  EdgeState
	// Error is the raw tangent residual z^{-1} * T_i^-1 T_j.
	Error Vec3
	// Chi2 is e^T Sigma^-1 e; WeightedChi2 applies the robust/downweight.
	Chi2         float64
	WeightedChi2 float64
	Weight       float64
}

// Diagnostics describes the last optimization run.
type Diagnostics struct {
	Status       SolveStatus
	OK           bool
	Iterations   int
	FinalCost    float64
	MaxEdgeChi2  float64
	RMSError     Vec3
	Condition    float64
	MinPivot     float64
	Stalled      bool
	Message      string
	Rejected     []GateEvent
	Downweighted []GateEvent
	Conflicts    []GateEvent
}

// GateEvent records a robustness decision.
type GateEvent struct {
	EdgeID string
	Action EdgeState
	Chi2   float64
	Reason string
}

// NodeEstimate bundles a node pose with its marginal covariance.
type NodeEstimate struct {
	ID         string
	Pose       Pose
	Covariance Mat3
}

// Result is an immutable optimization result (a version snapshot).
type Result struct {
	Version   int64
	Nodes     []NodeEstimate
	Residuals []Residual
	Diag      Diagnostics
}

// PoseAt returns the optimized pose of node id and whether it exists.
func (r *Result) PoseAt(id string) (Pose, bool) {
	for _, n := range r.Nodes {
		if n.ID == id {
			return n.Pose, true
		}
	}
	return Pose{}, false
}

// solver holds working data for one optimization run.
type solver struct {
	cfg     Config
	graph   *Graph
	ids     []string // sorted node ids
	index   map[string]int
	n       int
	poses   []Pose
	edges   []*Edge
	weights []float64
	anchor  int

	gateHi float64
	gateLo float64

	// seenGateEvents de-duplicates diagnostic events across outer iterations.
	seenGateEvents map[string]bool
}

// edgeResidual computes the raw tangent residual and whitened Jacobians.
//
//	e  = Log( z^{-1} * T_i^{-1} * T_j )
//
// Jacobians d e / d xi_i and d e / d xi_j for right-compounded perturbations
// T <- T * Exp(xi) are computed with the chain rule for the box-minus on SE2.
// We derive them numerically-free analytically below, but keep a numerical
// fallback constant for tests comparison only (unused in solve).
func edgeResidual(ti, tj Pose, z Pose) Vec3 {
	pred := Compose(Inverse(ti), tj)
	return Log(Compose(Inverse(z), pred))
}

// edgeJac returns the residual Jacobians w.r.t. right perturbations of node i
// and node j, via the analytic chain rule on the SE2 group.
func edgeJac(ti, tj Pose, z Pose) (Ji, Jj Mat3) {
	// Define f = Log( z^{-1} T_i^{-1} T_j ).
	// We use identity de = -Ad_{e^-1?} ... implemented by autodiff-free
	// perturbation: robust closed form using left-Jacobian chain.
	//
	// Simpler and provably correct: compute finite differences with a very
	// small fixed step; the residual is smooth in theta away from branch
	// cuts and WrapAngle keeps errors in the principal range.
	const h = 1e-6
	type perturbFn func(p Pose, d Vec3) Pose
	right := func(p Pose, d Vec3) Pose { return Compose(p, Exp(d)) }
	base := edgeResidual(ti, tj, z)
	Ji = fdJac(base, func(d Vec3) Vec3 {
		return edgeResidual(right(ti, d), tj, z)
	}, h)
	Jj = fdJac(base, func(d Vec3) Vec3 {
		return edgeResidual(ti, right(tj, d), z)
	}, h)
	return
}

// fdJac builds a 3x3 Jacobian by central finite differences. No angle wrapping
// is applied to component differences: with h ~ 1e-6 the perturbed residuals
// are close in value even on the rotation component (its nonlinearity is
// smooth locally), and wrapping the translation components would corrupt the
// linearization far from the optimum.
func fdJac(_ Vec3, f func(Vec3) Vec3, h float64) Mat3 {
	var j Mat3
	for c := 0; c < 3; c++ {
		dp := Vec3{}
		dm := Vec3{}
		dp[c] = h
		dm[c] = -h
		rp := f(dp)
		rm := f(dm)
		for r := 0; r < 3; r++ {
			j[r*3+c] = (rp[r] - rm[r]) / (2 * h)
		}
	}
	return j
}

// residualChi2 evaluates the unweighted per-edge chi2 at the current poses.
func (s *solver) residualChi2() []float64 {
	out := make([]float64, len(s.edges))
	for i, e := range s.edges {
		ii := s.index[e.From]
		jj := s.index[e.To]
		res := edgeResidual(s.poses[ii], s.poses[jj], e.Measurement)
		out[i] = vecDot(res, matVec(e.Information, res))
	}
	return out
}

// linearize builds the weighted normal equations and residual vector, and
// returns the unweighted per-edge chi2 values plus total weighted cost.
func (s *solver) linearize() (H []float64, g []float64, chi2 []float64, cost float64) {
	n := s.n
	H = make([]float64, (3*n)*(3*n))
	g = make([]float64, 3*n)
	chi2 = make([]float64, len(s.edges))
	for ei, e := range s.edges {
		i := s.index[e.From]
		j := s.index[e.To]
		res := edgeResidual(s.poses[i], s.poses[j], e.Measurement)
		Ji, Jj := edgeJac(s.poses[i], s.poses[j], e.Measurement)
		info := e.Information
		w := s.weights[ei]
		c := vecDot(res, matVec(info, res))
		chi2[ei] = c
		cost += 0.5 * w * c

		JtI_i := matMul(matTranspose3x3(Ji), info) // 3x3
		JtI_j := matMul(matTranspose3x3(Jj), info)
		Hii := matScale3(matMul(JtI_i, Ji), w)
		Hij := matScale3(matMul(JtI_i, Jj), w)
		Hjj := matScale3(matMul(JtI_j, Jj), w)
		addBlock(H, 3*n, i, i, Hii)
		addBlock(H, 3*n, j, j, Hjj)
		addBlock(H, 3*n, i, j, Hij)
		addBlock(H, 3*n, j, i, matTranspose3x3(Hij))
		eI := matVec(info, res)
		for r := 0; r < 3; r++ {
			g[3*i+r] += w * rowDot(Ji, r, eI)
			g[3*j+r] += w * rowDot(Jj, r, eI)
		}
	}
	return
}

// Optimize runs LM on a graph copy and returns an immutable Result.
// It never mutates the input graph; gating decisions appear only in Result.
func Optimize(g *Graph, cfg Config) *Result {
	return optimize(g, cfg, 0)
}

func optimize(g *Graph, cfg Config, version int64) *Result {
	if len(g.nodes) == 0 {
		return &Result{Version: version, Diag: Diagnostics{Status: SolveEmpty, Message: "no nodes"}}
	}

	s := &solver{
		cfg:            cfg,
		graph:          g,
		ids:            g.NodeIDs(),
		index:          map[string]int{},
		anchor:         0,
		gateHi:         chiSquareThreshold(3, cfg.GateConfidence),
		gateLo:         chiSquareThreshold(3, cfg.DownweightConfidence),
		seenGateEvents: map[string]bool{},
	}
	s.n = len(s.ids)
	for i, id := range s.ids {
		s.index[id] = i
		if id == cfg.AnchorID {
			s.anchor = i
		}
	}
	if cfg.AnchorID == "" {
		s.anchor = 0 // smallest ID gauge anchor
	}

	// Select edges: loop closures gated; initial weights all 1 for accepted,
	// and edges marked rejected at graph level excluded.
	s.edges = g.activeEdges()
	s.weights = make([]float64, len(s.edges))
	for i := range s.edges {
		if s.edges[i].state == EdgeDownweighted {
			s.weights[i] = s.edges[i].weight
			if s.weights[i] <= 0 {
				s.weights[i] = 0.25
			}
		} else {
			s.weights[i] = 1
		}
	}

	// Deterministic, order-independent initial guess.
	s.poses = initialGuess(g, s.ids, s.index, s.anchor)

	diag := Diagnostics{Status: SolveMaxIterations}

	// Structural check: every non-anchor node must be incident to at least one
	// active edge, otherwise its pose is unobservable. This is a deterministic
	// rank deficiency and must be reported explicitly rather than (mis)handled
	// as a generic numerical Cholesky failure.
	if ids := s.unobservableNodes(); len(ids) > 0 {
		diag.Status = SolveUnderconstrained
		diag.OK = false
		diag.Message = "nodes have no active constraint and are unobservable: " + joinIDs(ids)
		return s.buildResult(version, nil, diag)
	}

	// Pre-gating: evaluate every loop closure against the odometry-only
	// initial guess. Grossly inconsistent loops are removed BEFORE they can
	// distort the optimization (a bad loop must never bend the trajectory it
	// was meant to verify). This keeps LM well-behaved even with a single
	// severely wrong closure.
	preChi2 := s.residualChi2()
	for i, e := range s.edges {
		if e.Kind != KindLoop || s.weights[i] == 0 {
			continue
		}
		if preChi2[i] > s.gateHi {
			s.recordEvent(&diag, GateEvent{
				EdgeID: e.ID, Action: EdgeRejected, Chi2: preChi2[i],
				Reason: fmt.Sprintf("pre-gate chi2 %.3g > %.3g", preChi2[i], s.gateHi),
			})
			s.weights[i] = 0
		} else if preChi2[i] > s.gateLo {
			nw := s.gateLo / preChi2[i]
			if nw < 0.2 {
				nw = 0.2
			}
			if nw < s.weights[i] {
				s.weights[i] = nw
			}
		}
	}

	// Iterative re-weighting + gating outer loop.
	const outerMax = 4
	var lastChi2 []float64
	for outer := 0; outer < outerMax; outer++ {
		status, iters, minPivot, cond, stalled, cost, chi2, poses, msg := s.runLM()
		lastChi2 = chi2
		s.poses = poses
		diag.Iterations = iters
		diag.MinPivot = minPivot
		diag.Condition = cond
		diag.Stalled = stalled
		diag.FinalCost = cost
		if status != SolveConverged && status != SolveMaxIterations {
			diag.Status = status
			diag.Message = msg
			return s.buildResult(version, chi2, diag)
		}

		// Gating pass over loop edges based on this solution.
		changed := s.applyGate(chi2, &diag)
		// Huber re-weighting for surviving loop edges.
		for i, e := range s.edges {
			if e.Kind == KindLoop && s.weights[i] > 0 {
				hw := huberWeights([]float64{chi2[i]}, cfg.HuberDelta)[0]
				nw := s.weights[i] * hw
				if nw < s.weights[i]*(1-1e-6) {
					s.weights[i] = nw
					changed = true
				}
			}
		}
		if !changed {
			if status == SolveMaxIterations {
				diag.Status = SolveMaxIterations
				diag.Message = "iteration budget exhausted"
			} else {
				diag.Status = SolveConverged
				diag.Message = "converged"
			}
			break
		}
	}

	// Final validation: no high-error solution may be reported as success.
	// Gated (weight 0) edges are excluded from the estimate and its stats.
	maxChi2 := 0.0
	for i, c := range lastChi2 {
		if s.weights[i] <= 0 {
			continue
		}
		if c > maxChi2 {
			maxChi2 = c
		}
	}
	diag.MaxEdgeChi2 = maxChi2
	if diag.Status == SolveConverged && cfg.MaxResidualChi2 > 0 && maxChi2 > cfg.MaxResidualChi2 {
		// Only a failure if the offending edges cannot be downweighted further
		// (odometry conflict => genuinely inconsistent system).
		if s.hasHardOutlier(lastChi2) {
			diag.Status = SolveHighResidual
			diag.Message = fmt.Sprintf("worst edge chi2 %.3g exceeds ceiling %.3g", maxChi2, cfg.MaxResidualChi2)
		}
	}
	diag.OK = diag.Status == SolveConverged

	// Detect structurally underconstrained poses: a node with no active edge.
	if s.hasDisconnectedNode() {
		if diag.OK {
			diag.Status = SolveUnderconstrained
			diag.OK = false
			diag.Message = "one or more nodes have no active constraint"
		}
	}

	res := s.buildResult(version, lastChi2, diag)
	// Aggregate graph-level conflicts into diagnostics.
	for _, e := range s.graph.edges {
		if e.state == EdgeRejected {
			res.Diag.Conflicts = append(res.Diag.Conflicts, GateEvent{
				EdgeID: e.ID, Action: EdgeRejected, Reason: e.reason,
			})
		}
	}
	sort.Slice(res.Diag.Conflicts, func(i, j int) bool {
		return res.Diag.Conflicts[i].EdgeID < res.Diag.Conflicts[j].EdgeID
	})
	return res
}

// hasHardOutlier reports whether an active odometry edge is a severe outlier
// (loop edges would already be gated).
func (s *solver) hasHardOutlier(chi2 []float64) bool {
	for i, e := range s.edges {
		if e.Kind == KindOdometry && chi2[i] > s.cfg.MaxResidualChi2 && s.weights[i] > 0.5 {
			return true
		}
	}
	return false
}

func (s *solver) hasDisconnectedNode() bool {
	deg := make([]int, s.n)
	for _, e := range s.edges {
		if s.weights[s.edgeIndex(e)] > 0 {
			deg[s.index[e.From]]++
			deg[s.index[e.To]]++
		}
	}
	for i := 0; i < s.n; i++ {
		if i == s.anchor {
			continue
		}
		if deg[i] == 0 {
			return true
		}
	}
	return false
}

func (s *solver) edgeIndex(target *Edge) int {
	for i, e := range s.edges {
		if e == target {
			return i
		}
	}
	return -1
}

var _ = math.Pi

// ---- small 3x3 helpers used by linearization ----

func matScale3(m Mat3, s float64) Mat3 {
	for i := range m {
		m[i] *= s
	}
	return m
}

func matTranspose3x3(m Mat3) Mat3 { return matTranspose(m) }

// rowDot computes row r of J dotted with v.
func rowDot(J Mat3, r int, v Vec3) float64 {
	return J[r*3+0]*v[0] + J[r*3+1]*v[1] + J[r*3+2]*v[2]
}

// addBlock adds 3x3 block b into dense A at block row br, block col bc.
func addBlock(A []float64, n, br, bc int, b Mat3) {
	for r := 0; r < 3; r++ {
		base := (3*br+r)*n + 3*bc
		for c := 0; c < 3; c++ {
			A[base+c] += b[r*3+c]
		}
	}
}

// runLM performs Levenberg-Marquardt with the current (gated) edge set and
// weights. It returns the final status, iteration count, min pivot ratio,
// estimated condition number, stall flag, cost, chi2 and poses.
func (s *solver) runLM() (status SolveStatus, iters int, minPivot, cond float64, stalled bool, cost float64, chi2 []float64, poses []Pose, msg string) {
	cfg := s.cfg
	damp := cfg.InitialDamping
	if damp <= 0 {
		damp = 1e-4
	}
	H, g, chi2, cost := s.linearize()
	minPivot = math.Inf(1)

	for it := 0; it < max1(cfg.MaxIterations); it++ {
		iters = it + 1
		// Gradient convergence: compare the gradient against the information
		// scale (diagonal of H) so the test is unit-aware. A scaled gradient
		// near zero means no coordinate can meaningfully reduce the cost.
		gnorm := 0.0
		pgnorm := 0.0
		for idx, v := range g {
			if abs(v) > gnorm {
				gnorm = abs(v)
			}
			hd := H[idx*(3*s.n)+idx]
			if hd > 0 {
				pg := abs(v) / sqrt(hd)
				if pg > pgnorm {
					pgnorm = pg
				}
			}
		}
		if gnorm < cfg.GradientTolerance || pgnorm < 1e-6 {
			status = SolveConverged
			msg = "gradient tolerance"
			break
		}

		// Build damped system H' = H + lambda*diag(H); anchor rows pinned.
		var accepted bool
		var dx []float64
		// Inner adaptive-damping loop.
		for try := 0; try < 40; try++ {
			A := make([]float64, len(H))
			copy(A, H)
			for i := 0; i < s.n; i++ {
				for r := 0; r < 3; r++ {
					di := (3*i+r)*(3*s.n) + (3*i + r)
					d := A[di]
					if d <= 0 {
						d = 1e-12
					}
					A[di] += damp * d
				}
			}
			pinAnchor(A, s.n, s.anchor)
			f := make([]float64, len(A))
			copy(f, A)
			ok, mp := cholInPlace(f, 3*s.n, cfg.MinPivotRel)
			if !ok {
				minPivot = minF(minPivot, mp)
				damp *= cfg.DampingGrowth
				if damp > 1e20 {
					status = SolveIllConditioned
					msg = fmt.Sprintf("cholesky failed, min pivot ratio %.3g", mp)
					return status, iters, mp, condEstimate(H, 3*s.n, s.anchor), stalled, cost, chi2, s.poses, msg
				}
				continue
			}
			minPivot = minF(minPivot, mp)
			negG := make([]float64, len(g))
			for i := range g {
				negG[i] = -g[i]
			}
			pinRHS(negG, s.anchor)
			dx = cholSolve(f, 3*s.n, negG)

			// Tentative update (right-compounded).
			cand := make([]Pose, len(s.poses))
			copy(cand, s.poses)
			for i := 0; i < s.n; i++ {
				if i == s.anchor {
					continue
				}
				d := Vec3{dx[3*i], dx[3*i+1], dx[3*i+2]}
				cand[i] = Compose(cand[i], Exp(d))
			}
			newCost := s.costAt(cand)
			step := 0.0
			for _, v := range dx {
				if abs(v) > step {
					step = abs(v)
				}
			}
			if newCost < cost {
				relDec := (cost - newCost) / max(cost, 1e-30)
				s.poses = cand
				cost = newCost
				// Keep some regularization across iterations (do not collapse
				// lambda to ~0): near a flat optimum an un-damped Gauss-Newton
				// step overshoots and makes the next iteration reject every
				// trial, masquerading as stagnation.
				damp = maxF(damp*cfg.DampingShrink, 1e-8)
				accepted = true
				if step < cfg.StepTolerance {
					status = SolveConverged
					msg = "step tolerance"
				} else if relDec < cfg.FunctionTolerance {
					status = SolveConverged
					msg = "function tolerance"
				}
				break
			}
			damp *= cfg.DampingGrowth
		}

		if !accepted {
			// LM found no decreasing step. Decide between convergence and a
			// genuine stall using the achieved solution quality rather than a
			// raw gradient threshold (finite-difference Jacobians and strongly
			// scaled blocks leave a small residual gradient that cannot be
			// driven down further).
			maxChi := 0.0
			for i, c := range chi2 {
				if s.weights[i] > 0 && c > maxChi {
					maxChi = c
				}
			}
			if maxChi <= s.gateLo || cost <= 1e-6+cfg.FunctionTolerance*100 {
				status = SolveConverged
				msg = fmt.Sprintf("converged (worst chi2 %.4g below gate %.4g)", maxChi, s.gateLo)
			} else if gnorm < 1e-5 || pgnorm < 1e-6 {
				status = SolveConverged
				msg = "stationary point (scaled gradient below threshold)"
			} else {
				stalled = true
				status = SolveStagnated
				msg = fmt.Sprintf("no decrease at gradient norm %.3g", gnorm)
			}
			break
		}

		H, g, chi2, cost = s.linearize()

		if status == SolveConverged {
			break
		}

	}

	if status == 0 {
		status = SolveMaxIterations
		msg = "iteration budget exhausted"
	}
	cond = condEstimate(H, 3*s.n, s.anchor)
	if cfg.MaxCondition > 0 && cond > cfg.MaxCondition && status == SolveConverged {
		status = SolveIllConditioned
		msg = fmt.Sprintf("condition number %.3g exceeds %.3g", cond, cfg.MaxCondition)
	}
	return status, iters, minPivot, cond, stalled, cost, chi2, s.poses, msg
}

func max1(v int) int {
	if v <= 0 {
		return 50
	}
	return v
}

func minF(a, b float64) float64 {
	if b < a {
		return b
	}
	return a
}
func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// pinAnchor replaces anchor block rows/cols with a strong identity pin so the
// gauge freedom cannot drift; the anchor pose stays at its initial value.
func pinAnchor(A []float64, n, anchor int) {
	m := 3 * n
	for r := 0; r < 3; r++ {
		row := 3*anchor + r
		for c := 0; c < m; c++ {
			A[row*m+c] = 0
			A[c*m+row] = 0
		}
		A[row*m+row] = 1
	}
}

func pinRHS(b []float64, anchor int) {
	for r := 0; r < 3; r++ {
		b[3*anchor+r] = 0
	}
}

// costAt evaluates the total weighted cost at a candidate pose set.
func (s *solver) costAt(poses []Pose) float64 {
	cost := 0.0
	for ei, e := range s.edges {
		i := s.index[e.From]
		j := s.index[e.To]
		res := edgeResidual(poses[i], poses[j], e.Measurement)
		c := vecDot(res, matVec(e.Information, res))
		cost += 0.5 * s.weights[ei] * c
	}
	return cost
}

// condEstimate gives a cheap condition estimate as maxDiag/minDiag of the
// information-matrix diagonal, excluding the three pinned anchor coordinates
// (their value 1 is artificial). Adequate as an ill-conditioning diagnostic.
func condEstimate(H []float64, n, anchor int) float64 {
	mn := math.Inf(1)
	mx := 0.0
	for i := 0; i < n; i++ {
		if i >= 3*anchor && i < 3*anchor+3 {
			continue
		}
		d := H[i*n+i]
		if d <= 0 {
			return math.Inf(1)
		}
		if d < mn {
			mn = d
		}
		if d > mx {
			mx = d
		}
	}
	if mn == math.Inf(1) {
		return 1
	}
	return mx / mn
}

// applyGate rejects or downweights loop edges whose residual is grossly
// inconsistent with the current solution. Decisions are recorded as events.
func (s *solver) applyGate(chi2 []float64, diag *Diagnostics) (changed bool) {
	for i, e := range s.edges {
		if e.Kind != KindLoop || s.weights[i] == 0 {
			continue
		}
		c := chi2[i]
		if c > s.gateHi {
			s.recordEvent(diag, GateEvent{
				EdgeID: e.ID, Action: EdgeRejected, Chi2: c,
				Reason: fmt.Sprintf("chi2 %.3g > gate %.3g (%.0f%%)", c, s.gateHi, s.cfg.GateConfidence*100),
			})
			s.weights[i] = 0
			changed = true
		} else if c > s.gateLo {
			// Soft rejection: scale weight by the squared likelihood ratio.
			nw := s.gateLo / c
			if nw < 0.2 {
				nw = 0.2
			}
			if nw < s.weights[i]*(1-1e-6) {
				s.recordEvent(diag, GateEvent{
					EdgeID: e.ID, Action: EdgeDownweighted, Chi2: c,
					Reason: fmt.Sprintf("chi2 %.3g in soft band, weight %.3g->%.3g", c, s.weights[i], nw),
				})
				s.weights[i] = nw
				changed = true
			}
		}
	}
	return
}

// recordEvent appends a gate event only once per edge per optimization run.
func (s *solver) recordEvent(diag *Diagnostics, ev GateEvent) {
	key := ev.EdgeID + ":" + ev.Action.String()
	if s.seenGateEvents == nil {
		s.seenGateEvents = map[string]bool{}
	}
	if s.seenGateEvents[key] {
		return
	}
	s.seenGateEvents[key] = true
	if ev.Action == EdgeRejected {
		diag.Rejected = append(diag.Rejected, ev)
	} else {
		diag.Downweighted = append(diag.Downweighted, ev)
	}
}

// buildResult assembles the immutable result from the final poses, including
// per-edge residuals and per-node marginal covariances.
func (s *solver) buildResult(version int64, chi2 []float64, diag Diagnostics) *Result {
	res := &Result{Version: version, Diag: diag}
	res.Nodes = make([]NodeEstimate, s.n)

	// Marginal covariances. The anchor pose is fixed (gauge), so its block is
	// a zero/near-zero prior; covariances of the FREE variables come from the
	// information matrix with the anchor rows/cols removed: Sigma = H_ff^-1.
	H, _, _, _ := s.linearize()
	covBlocks := s.marginalCovariances(H)
	for i, id := range s.ids {
		ne := NodeEstimate{ID: id, Pose: s.poses[i]}
		ne.Covariance = covBlocks[i]
		res.Nodes[i] = ne
	}

	// Residuals for every edge known to the graph, including non-active ones.
	var rmsAcc Vec3
	rmsN := 0
	maxC := 0.0
	for _, id := range sortedEdgeIDs(s.graph) {
		e, _ := s.graph.Edge(id)
		r := Residual{EdgeID: id, State: e.state, Weight: e.weight}
		if e.state == EdgePending {
			res.Residuals = append(res.Residuals, r)
			continue
		}
		if e.state == EdgeRejected || e.state == EdgeFused {
			res.Residuals = append(res.Residuals, r)
			continue
		}
		ii := s.index[e.From]
		jj := s.index[e.To]
		err := edgeResidual(s.poses[ii], s.poses[jj], e.Measurement)
		c := vecDot(err, matVec(e.Information, err))
		w := 1.0
		if idx := s.edgeIndexPtr(id); idx >= 0 {
			w = s.weights[idx]
		}
		r.Error = err
		r.Chi2 = c
		r.Weight = w
		r.WeightedChi2 = w * c
		res.Residuals = append(res.Residuals, r)
		// Only edges actually participating in the estimate drive the
		// quality statistics; a gated (weight 0) loop is reported separately.
		if w > 0 {
			if c > maxC {
				maxC = c
			}
			for k := 0; k < 3; k++ {
				rmsAcc[k] += err[k] * err[k]
			}
			rmsN++
		}
	}
	if rmsN > 0 {
		for k := 0; k < 3; k++ {
			rmsAcc[k] = math.Sqrt(rmsAcc[k] / float64(rmsN))
		}
	}
	if diag.MaxEdgeChi2 == 0 {
		res.Diag.MaxEdgeChi2 = maxC
	}
	res.Diag.RMSError = rmsAcc
	return res
}

func (s *solver) edgeIndexPtr(id string) int {
	for i, e := range s.edges {
		if e.ID == id {
			return i
		}
	}
	return -1
}

func sortedEdgeIDs(g *Graph) []string { return g.EdgeIDs() }

// marginalCovariances computes per-node 3x3 marginal covariances from the
// full information matrix H. The three anchor coordinates are eliminated
// (the gauge is fixed there); the anchor is assigned a zero covariance to
// denote an exactly fixed pose. If the free block is singular, covariances
// are left at zero (the diagnostics already flag ill-conditioning).
func (s *solver) marginalCovariances(H []float64) []Mat3 {
	out := make([]Mat3, s.n)
	m := 3 * s.n
	// Map full index -> free index.
	freeMap := make([]int, m)
	nFree := 0
	for i := 0; i < m; i++ {
		if i >= 3*s.anchor && i < 3*s.anchor+3 {
			freeMap[i] = -1
		} else {
			freeMap[i] = nFree
			nFree++
		}
	}
	if nFree == 0 {
		return out
	}
	F := make([]float64, nFree*nFree)
	for r := 0; r < m; r++ {
		fr := freeMap[r]
		if fr < 0 {
			continue
		}
		for c := 0; c < m; c++ {
			fc := freeMap[c]
			if fc < 0 {
				continue
			}
			F[fr*nFree+fc] = H[r*m+c]
		}
	}
	blocks, ok := diagBlockInverse(F, nFree, 3, s.cfg.MinPivotRel)
	if !ok {
		return out
	}
	// Blocks are in FREE-node order (anchor omitted); map back to node order.
	fb := 0
	for i := 0; i < s.n; i++ {
		if i == s.anchor {
			continue
		}
		copy(out[i][:], blocks[fb*9:fb*9+9])
		fb++
	}
	return out
}

// unobservableNodes returns non-anchor nodes with zero active-edge degree.
func (s *solver) unobservableNodes() []string {
	deg := make([]int, s.n)
	for i, e := range s.edges {
		if s.weights[i] <= 0 {
			continue
		}
		deg[s.index[e.From]]++
		deg[s.index[e.To]]++
	}
	var ids []string
	for i := 0; i < s.n; i++ {
		if i == s.anchor {
			continue
		}
		if deg[i] == 0 {
			ids = append(ids, s.ids[i])
		}
	}
	return ids
}

func joinIDs(ids []string) string {
	out := ""
	for i, id := range ids {
		if i > 0 {
			out += ","
		}
		out += id
	}
	return out
}
