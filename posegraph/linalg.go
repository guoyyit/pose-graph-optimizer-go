package posegraph

import (
	"errors"
	"math"
)

// Dense linear algebra over flat row-major []float64 matrices. Systems handed
// here are symmetric (normal equations of a least-squares problem) and are
// expected to be positive definite; rank deficiency is reported rather than
// answered with a bogus solution.

var errNotSPD = errors.New("posegraph: matrix is not symmetric positive definite")

func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// cholesky computes the lower-triangular factor L with A = L Lᵀ.
// A zero/negative pivot or a non-finite entry yields errNotSPD.
func cholesky(a []float64, n int) ([]float64, error) {
	l := make([]float64, n*n)
	for j := 0; j < n; j++ {
		for i := j; i < n; i++ {
			sum := a[i*n+j]
			if !isFinite(sum) {
				return nil, errNotSPD
			}
			for k := 0; k < j; k++ {
				sum -= l[i*n+k] * l[j*n+k]
			}
			if i == j {
				if sum <= 0 || !isFinite(sum) {
					return nil, errNotSPD
				}
				l[i*n+j] = math.Sqrt(sum)
			} else {
				l[i*n+j] = sum / l[j*n+j]
			}
		}
	}
	return l, nil
}

// solveCholesky solves L Lᵀ x = b given lower-triangular L. b is copied.
func solveCholesky(l []float64, n int, b []float64) []float64 {
	y := make([]float64, n)
	for i := 0; i < n; i++ { // forward solve L y = b
		sum := b[i]
		for j := 0; j < i; j++ {
			sum -= l[i*n+j] * y[j]
		}
		y[i] = sum / l[i*n+i]
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- { // back solve Lᵀ x = y
		sum := y[i]
		for j := i + 1; j < n; j++ {
			sum -= l[j*n+i] * x[j]
		}
		x[i] = sum / l[i*n+i]
	}
	return x
}

// inverseCholesky computes A⁻¹ column by column from factor L.
func inverseCholesky(l []float64, n int) []float64 {
	inv := make([]float64, n*n)
	e := make([]float64, n)
	for col := 0; col < n; col++ {
		for i := range e {
			e[i] = 0
		}
		e[col] = 1
		x := solveCholesky(l, n, e)
		for row := 0; row < n; row++ {
			inv[row*n+col] = x[row]
		}
	}
	return inv
}

// reciprocalCondition estimates 1/cond(A) in the 2-norm from L's pivots:
// cond₂(A) ≥ (max L_ii / min L_ii)², so (min/max)² is a conservative proxy.
// Returns 0 when no positive pivot exists.
func reciprocalCondition(l []float64, n int) float64 {
	minP, maxP := math.Inf(1), 0.0
	for i := 0; i < n; i++ {
		p := l[i*n+i]
		if p <= 0 || !isFinite(p) {
			return 0
		}
		if p < minP {
			minP = p
		}
		if p > maxP {
			maxP = p
		}
	}
	if math.IsInf(minP, 1) || maxP == 0 {
		return 0
	}
	r := minP / maxP
	return r * r
}
