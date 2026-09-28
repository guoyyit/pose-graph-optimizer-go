package posegraph

import "math"

// Dense symmetric linear algebra for the optimizer normal equations.
// Matrices are row-major n*n slices; only the symmetric part is read.

// cholInPlace computes a lower-triangular Cholesky factor L (stored in the
// lower triangle of a; the upper triangle is zeroed) for symmetric positive
// definite A. Pivot validity is scale-aware: the reduced pivot is compared
// against the original diagonal magnitude of that same coordinate times
// relTol, so strongly (but legitimately) scaled blocks cannot be mistaken
// for rank deficiency. It reports the smallest pivot-to-diagonal ratio.
func cholInPlace(a []float64, n int, relTol float64) (bool, float64) {
	origDiag := make([]float64, n)
	maxDiag := 0.0
	for i := 0; i < n; i++ {
		origDiag[i] = a[i*n+i]
		if d := a[i*n+i]; d > maxDiag {
			maxDiag = d
		}
	}
	if maxDiag <= 0 {
		return false, 0
	}
	minRatio := math.Inf(1)
	for j := 0; j < n; j++ {
		// diagonal element
		s := a[j*n+j]
		for k := 0; k < j; k++ {
			s -= a[j*n+k] * a[j*n+k]
		}
		// Scale-aware floor: tiny relative to THIS coordinate's own
		// information (not the global max), or non-positive => failure.
		floor := relTol * origDiag[j]
		if s <= floor {
			return false, minRatio
		}
		if origDiag[j] > 0 {
			if r := s / origDiag[j]; r < minRatio {
				minRatio = r
			}
		}
		ljj := math.Sqrt(s)
		a[j*n+j] = ljj
		for i := j + 1; i < n; i++ {
			t := a[i*n+j]
			for k := 0; k < j; k++ {
				t -= a[i*n+k] * a[j*n+k]
			}
			a[i*n+j] = t / ljj
			// Keep the upper triangle from holding stale A entries: only the
			// lower triangle is L, and cholSolve must read L, never A.
			a[j*n+i] = 0
		}
	}
	return true, minRatio
}

// cholSolve solves (L L^T) x = b given the in-place factor from cholInPlace.
// b is unchanged; a fresh slice is returned.
func cholSolve(a []float64, n int, b []float64) []float64 {
	x := make([]float64, n)
	// forward solve L y = b
	for i := 0; i < n; i++ {
		s := b[i]
		for k := 0; k < i; k++ {
			s -= a[i*n+k] * x[k]
		}
		x[i] = s / a[i*n+i]
	}
	// back solve L^T x = y
	for i := n - 1; i >= 0; i-- {
		s := x[i]
		for k := i + 1; k < n; k++ {
			s -= a[k*n+i] * x[k]
		}
		x[i] = s / a[i*n+i]
	}
	return x
}

// diagBlockInverse computes selected diagonal blocks of A^-1 without forming
// the full inverse, by solving against identity columns via Cholesky.
// blockSize must divide n; the returned slice holds (n/blockSize) blocks of
// blockSize*blockSize in node order.
func diagBlockInverse(a []float64, n, blockSize int, relTol float64) ([]float64, bool) {
	f := make([]float64, n*n)
	copy(f, a)
	ok, _ := cholInPlace(f, n, relTol)
	if !ok {
		return nil, false
	}
	nb := n / blockSize
	out := make([]float64, nb*blockSize*blockSize)
	for bi := 0; bi < nb; bi++ {
		for c := 0; c < blockSize; c++ {
			col := bi*blockSize + c
			e := make([]float64, n)
			e[col] = 1
			x := cholSolve(f, n, e)
			for r := 0; r < blockSize; r++ {
				row := bi*blockSize + r
				// X_{row,col} of the inverse
				out[bi*blockSize*blockSize+r*blockSize+c] = x[row]
			}
		}
	}
	// Symmetrize each block.
	for bi := 0; bi < nb; bi++ {
		b := out[bi*blockSize*blockSize : bi*blockSize*blockSize+blockSize*blockSize]
		for i := 0; i < blockSize; i++ {
			for j := i + 1; j < blockSize; j++ {
				v := (b[i*blockSize+j] + b[j*blockSize+i]) / 2
				b[i*blockSize+j] = v
				b[j*blockSize+i] = v
			}
		}
	}
	return out, true
}
