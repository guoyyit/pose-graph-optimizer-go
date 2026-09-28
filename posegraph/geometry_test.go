package posegraph

import (
	"math"
	"testing"
)

func approx(v, want, tol float64) bool { return math.Abs(v-want) <= tol }

func TestComposeInverse(t *testing.T) {
	a := Pose2{1, 2, 0.7}
	b := Pose2{-0.5, 0.3, -1.2}
	c := compose(a, b)
	roundtrip := compose(inversePose(b), compose(inversePose(a), c))
	if !approx(roundtrip.X, 0, 1e-12) || !approx(roundtrip.Y, 0, 1e-12) || !approx(roundtrip.Theta, 0, 1e-12) {
		t.Fatalf("identity roundtrip failed: %+v", roundtrip)
	}
}

func numJacobians(pa, pb Pose2, z Pose2, eps float64) (ja, jb Mat3) {
	base := relativeError(z, between(pa, pb))
	for node := 0; node < 2; node++ {
		for d := 0; d < 3; d++ {
			ca, cb := pa, pb
			if node == 0 {
				if d == 0 {
					ca.X += eps
				}
				if d == 1 {
					ca.Y += eps
				}
				if d == 2 {
					ca.Theta += eps
				}
			} else {
				if d == 0 {
					cb.X += eps
				}
				if d == 1 {
					cb.Y += eps
				}
				if d == 2 {
					cb.Theta += eps
				}
			}
			e := relativeError(z, between(ca, cb))
			target := &ja
			if node == 1 {
				target = &jb
			}
			for r := 0; r < 3; r++ {
				val := (e[r] - base[r]) / eps
				if r == 2 && math.Abs(e[r]-base[r]) > math.Pi { // wrapped
				}
				target[r*3+d] = val
			}
		}
	}
	return ja, jb
}

func TestRelJacobians(t *testing.T) {
	cases := []struct{ pa, pb, z Pose2 }{
		{Pose2{0, 0, 0}, Pose2{1, 0.5, 0.3}, Pose2{1, 0.5, 0.3}},
		{Pose2{2, -1, 1.1}, Pose2{3, 0.2, 1.4}, Pose2{0.9, 1.3, 0.35}},
		{Pose2{-3, 0.5, -2.2}, Pose2{-1, -0.7, -1.9}, Pose2{2.1, 0.2, 0.25}},
	}
	for i, tc := range cases {
		ja, jb := relJacobiansWorld(tc.pa, tc.pb, tc.z)
		na, nb := numJacobians(tc.pa, tc.pb, tc.z, 1e-6)
		for r := 0; r < 3; r++ {
			for c := 0; c < 3; c++ {
				if math.Abs(ja[r*3+c]-na[r*3+c]) > 1e-5 {
					t.Errorf("case %d Ja[%d,%d]=%v want %v", i, r, c, ja[r*3+c], na[r*3+c])
				}
				if math.Abs(jb[r*3+c]-nb[r*3+c]) > 1e-5 {
					t.Errorf("case %d Jb[%d,%d]=%v want %v", i, r, c, jb[r*3+c], nb[r*3+c])
				}
			}
		}
	}
}

func TestPriorJacobian(t *testing.T) {
	z := Pose2{1, 2, 0.9}
	p := Pose2{1.5, 1.8, 1.2}
	base := relativeError(z, p)
	j := priorJacobianWorld(z)
	eps := 1e-6
	for d := 0; d < 3; d++ {
		cp := p
		if d == 0 {
			cp.X += eps
		}
		if d == 1 {
			cp.Y += eps
		}
		if d == 2 {
			cp.Theta += eps
		}
		e := relativeError(z, cp)
		for r := 0; r < 3; r++ {
			got := (e[r] - base[r]) / eps
			if math.Abs(got-j[r*3+d]) > 1e-5 {
				t.Errorf("prior J[%d,%d]=%v want %v", r, d, j[r*3+d], got)
			}
		}
	}
}
