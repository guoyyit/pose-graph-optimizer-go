package posegraph

import (
	"fmt"
	"math"
)

// gateLoop evaluates a loop-closure candidate against the current best
// estimate. It computes the whitened residual the candidate would induce
// WITHOUT adding it (so the candidate cannot vote for itself), then applies
// two deterministic rules:
//
//  1. chi2/DOF above cfg.LoopChi2Gate, or an unreasonably large raw
//     discrepancy, => reject (the candidate is kept on a rejected list and
//     never touches the estimate).
//  2. residual above the robust-kernel delta => accept down-weighted by a DCS
//     (dynamic covariance scaling) factor so a mildly bad loop cannot distort
//     the rest of the graph.
//  3. otherwise accept at full weight.
//
// Rejected loops leave the graph untouched, hence all other estimates remain
// available.
func gateLoop(cfg Config, g *Graph, current map[NodeID]Pose2, c Constraint) LoopDecision {
	d := LoopDecision{ConstraintID: c.ID}

	pa, oka := current[c.A]
	pb, okb := current[c.B]
	if !oka || !okb {
		d.Reason = "endpoints not estimated yet"
		return d
	}
	e := relativeError(c.Measurement, between(pa, pb))
	info, ok := covarianceInformation(c.Covariance, cfg.CovarianceFloor)
	if !ok {
		d.Reason = "invalid candidate covariance"
		return d
	}
	we := matVec3(info, e)
	chi2 := e[0]*we[0] + e[1]*we[1] + e[2]*we[2]
	dof := 3.0
	d.Chi2PerDOF = chi2 / dof

	const rawReject = 50.0 // metres / radians-scale hard inconsistency
	if math.Abs(e[0]) > rawReject || math.Abs(e[1]) > rawReject ||
		math.Abs(e[2]) > math.Pi {
		d.Reason = fmt.Sprintf("gross inconsistency dx=%.3f dy=%.3f dth=%.3f", e[0], e[1], e[2])
		return d
	}
	if d.Chi2PerDOF > cfg.LoopChi2Gate {
		d.Reason = fmt.Sprintf("chi2/DOF %.3f exceeds gate %.3f", d.Chi2PerDOF, cfg.LoopChi2Gate)
		return d
	}

	// Accepted. Down-weight if beyond the robust kernel; whitened norm is
	// sqrt(chi2).
	wn := math.Sqrt(math.Max(chi2, 0))
	wt := robustWeight(cfg, wn)
	d.Accepted = true
	d.Weight = wt
	if wt < 1 {
		d.Downweighted = true
		d.Reason = fmt.Sprintf("accepted with robust weight %.4f (whitened norm %.3f)", wt, wn)
	}
	return d
}

// robustWeight returns a DCS-style multiplier for a whitened residual norm.
//
//	norm <= delta : w = 1
//	norm >  delta : w = (2·delta)/(norm + delta/norm)
//
// This equals the classic DCS factor (delta/norm)² expressed in a numerically
// safer form, is C¹ at the threshold and drives influence toward zero as the
// residual grows. delta <= 0 disables robust weighting (w = 1).
func robustWeight(cfg Config, norm float64) float64 {
	delta := cfg.RobustKernel
	if delta <= 0 || norm <= delta {
		return 1
	}
	if norm <= 0 {
		return 1
	}
	w := 2 * delta / (norm + delta/norm)
	if w < 1e-6 {
		w = 1e-6
	}
	if w > 1 {
		w = 1
	}
	return w
}
