package posegraph

import "math"

// Robust statistics helpers: chi-square gate thresholds and Huber kernel.
// All routines are pure functions so gating decisions are deterministic.

// gammln returns ln(Gamma(x)) (Lanczos approximation, Numerical Recipes).
func gammln(x float64) float64 {
	cof := []float64{
		76.18009172947146, -86.50532032941677, 24.01409824083091,
		-1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
	}
	ser := 1.000000000190015
	y := x
	tmp := x + 5.5
	tmp -= (x + 0.5) * math.Log(tmp)
	for j := 0; j < 6; j++ {
		y++
		ser += cof[j] / y
	}
	return -tmp + math.Log(2.5066282746310005*ser/x)
}

// series computes the regularized lower incomplete gamma P(a,x) by series.
func gammaSeries(a, x float64) float64 {
	if x <= 0 {
		return 0
	}
	ap := a
	sum := 1 / a
	del := sum
	for n := 0; n < 200; n++ {
		ap++
		del *= x / ap
		sum += del
		if math.Abs(del) < 1e-14*math.Abs(sum) {
			break
		}
	}
	return sum * math.Exp(-x+a*math.Log(x)-gammln(a))
}

// cf computes the regularized upper incomplete gamma Q(a,x) by CF.
func gammaCF(a, x float64) float64 {
	b := x + 1 - a
	c := math.MaxFloat64
	d := 1 / b
	h := d
	for i := 1; i < 200; i++ {
		an := -float64(i) * (float64(i) - a)
		b += 2
		d = an*d + b
		if math.Abs(d) < 1e-300 {
			d = 1e-300
		}
		c = b + an/c
		if math.Abs(c) < 1e-300 {
			c = 1e-300
		}
		d = 1 / d
		delta := d * c
		h *= delta
		if math.Abs(delta-1) < 1e-14 {
			break
		}
	}
	return math.Exp(-x+a*math.Log(x)-gammln(a)) * h
}

// gammaP returns the regularized lower incomplete gamma P(a,x).
func gammaP(a, x float64) float64 {
	if x < 0 || a <= 0 {
		return 0
	}
	if x < a+1 {
		return gammaSeries(a, x)
	}
	return 1 - gammaCF(a, x)
}

// chiSquareCDF returns P(X <= x) for dof degrees of freedom.
func chiSquareCDF(x float64, dof int) float64 {
	if x <= 0 {
		return 0
	}
	return gammaP(float64(dof)/2, x/2)
}

// chiSquareThreshold inverts chiSquareCDF by bisection for the given dof.
// confidence must be in (0,1). Accuracy is ~1e-8 relative, plenty for gates.
func chiSquareThreshold(dof int, confidence float64) float64 {
	if confidence <= 0 {
		return 0
	}
	if confidence >= 1 {
		return math.Inf(1)
	}
	lo, hi := 0.0, 1.0
	for chiSquareCDF(hi, dof) < confidence {
		lo = hi
		hi *= 2
	}
	for i := 0; i < 80; i++ {
		mid := (lo + hi) / 2
		if chiSquareCDF(mid, dof) < confidence {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// huberWeights applies a Huber robust kernel to per-edge chi2 values and
// returns scalar weights in (0,1]. The standardized residual magnitude is
// r = sqrt(chi2); w = 1 for r <= delta, delta/r otherwise. delta <= 0
// disables the kernel.
func huberWeights(chi2PerEdge []float64, delta float64) []float64 {
	w := make([]float64, len(chi2PerEdge))
	for i, c := range chi2PerEdge {
		if delta <= 0 {
			w[i] = 1
			continue
		}
		r := math.Sqrt(math.Max(0, c))
		if r <= delta {
			w[i] = 1
		} else {
			w[i] = delta / r
		}
	}
	return w
}

// mergeMeasurements fuses relative measurements between the same ordered node
// pair by information-weighted averaging in the tangent space. ws are extra
// scalar multipliers (pass 1 for plain duplicate fusion). Returns the fused
// pose and the summed information (additive for independent measurements).
func mergeMeasurements(ms []Pose, infos []Mat3, ws []float64) (Pose, Mat3) {
	var acc Vec3
	var wsum float64
	var info Mat3
	for i, m := range ms {
		v := Log(m)
		// Tangent-space per-component information weighting: weight =
		// ws[i] * tr(Sigma^-1)/3, a rotation-independent scalar strength.
		w := ws[i] * (infos[i][0] + infos[i][4] + infos[i][8]) / 3
		if w <= 0 || math.IsNaN(w) {
			w = ws[i]
		}
		acc = vecAdd(acc, vecScale(v, w))
		wsum += w
		for k := 0; k < 9; k++ {
			info[k] += infos[i][k] * ws[i]
		}
	}
	if wsum <= 0 {
		wsum = float64(len(ms))
	}
	fused := Exp(vecScale(acc, 1/wsum))
	return fused, info
}
