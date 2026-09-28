package posegraph

import (
	"math"
	"testing"
)

func approxEq(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

func TestSE2ExpLogRoundTrip(t *testing.T) {
	cases := []Vec3{{0, 0, 0}, {1, 2, 0.3}, {-1, 0.5, -1.2}, {0.1, -0.2, 2.5}, {3, -2, -3}}
	for _, v := range cases {
		l := Log(Exp(v))
		for k := 0; k < 3; k++ {
			if math.Abs(l[k]-v[k]) > 1e-8 {
				t.Fatalf("log(exp(%v)) = %v", v, l)
			}
		}
	}
	for _, p := range []Pose{{0, 0, 0}, {1, -2, 0.9}, {-3, 1, -2}} {
		id := Compose(p, Inverse(p))
		approxEq(t, "inv x", id.X, 0, 1e-9)
		approxEq(t, "inv y", id.Y, 0, 1e-9)
		approxEq(t, "inv th", id.Theta, 0, 1e-9)
	}
}

func TestSE2Compose(t *testing.T) {
	// rotate 90deg then translate in local frame
	a := Pose{X: 1, Y: 1, Theta: math.Pi / 2}
	b := Pose{X: 2, Y: 0, Theta: 0}
	c := Compose(a, b)
	approxEq(t, "cx", c.X, 1, 1e-9)
	approxEq(t, "cy", c.Y, 3, 1e-9)
	approxEq(t, "cth", c.Theta, math.Pi/2, 1e-9)
}

func TestWrapAngle(t *testing.T) {
	cases := map[float64]float64{3 * math.Pi: -math.Pi, -3 * math.Pi: -math.Pi, 0.5: 0.5}
	for in, want := range cases {
		got := WrapAngle(in)
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("wrap(%v)=%v want %v", in, got, want)
		}
	}
}

func TestChiSquareThresholds(t *testing.T) {
	cases := []struct {
		cf   float64
		want float64
	}{{0.5, 2.366}, {0.95, 7.815}, {0.99, 11.345}}
	for _, c := range cases {
		got := chiSquareThreshold(3, c.cf)
		if math.Abs(got-c.want) > 0.02 {
			t.Errorf("chi2(%v)=%v want %v", c.cf, got, c.want)
		}
		// CDF/quantile consistency
		p := chiSquareCDF(got, 3)
		if math.Abs(p-c.cf) > 1e-4 {
			t.Errorf("cdf(quantile) mismatch: %v", p)
		}
	}
}
