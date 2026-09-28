package posegraph

import "math"

// SE(2) geometry. A pose maps body-frame points to world frame by
//	x_w = R(theta) x_b + t, R(theta) = [[c,-s],[s,c]].
// All stored angles are wrapped to (-pi, pi].

func normalizeAngle(a float64) float64 {
	for a > math.Pi {
		a -= 2 * math.Pi
	}
	for a <= -math.Pi {
		a += 2 * math.Pi
	}
	return a
}

// compose returns a ⊕ b: T_a * T_b.
func compose(a, b Pose2) Pose2 {
	ca, sa := math.Cos(a.Theta), math.Sin(a.Theta)
	return Pose2{
		X:     a.X + ca*b.X - sa*b.Y,
		Y:     a.Y + sa*b.X + ca*b.Y,
		Theta: normalizeAngle(a.Theta + b.Theta),
	}
}

// inversePose returns a⁻¹.
func inversePose(a Pose2) Pose2 {
	ca, sa := math.Cos(a.Theta), math.Sin(a.Theta)
	return Pose2{
		X:     -ca*a.X - sa*a.Y,
		Y:     sa*a.X - ca*a.Y,
		Theta: normalizeAngle(-a.Theta),
	}
}

// between returns a⁻¹ ⊕ b.
func between(a, b Pose2) Pose2 { return compose(inversePose(a), b) }

// relativeError returns e = measured⁻¹ ⊕ actual in R^3 with wrapped angle.
func relativeError(measured, actual Pose2) Vec3 {
	e := compose(inversePose(measured), actual)
	return Vec3{e.X, e.Y, e.Theta}
}

// relJacobiansWorld returns the 3x3 error Jacobians d(e)/d(pa), d(e)/d(pb) for
// a relative constraint z with e = z⁻¹ ⊕ (pa⁻¹ ⊕ pb), using world-frame
// perturbations [dx, dy, dtheta] applied directly to each pose.
//
// Let h = pa⁻¹ ⊕ pb, R = R(-thetaA-thetaZ) and v = -Rzᵀ J h.t:
//
//	Jb = [[ R      0 ],   Ja = [[ -R     v ],
//	      [ 0 0    1 ]]        [  0 0   -1 ]]
func relJacobiansWorld(pa, pb Pose2, z Pose2) (ja, jb Mat3) {
	h := between(pa, pb)
	chx, shy := math.Cos(-pa.Theta-z.Theta), math.Sin(-pa.Theta-z.Theta)
	cz, sz := math.Cos(z.Theta), math.Sin(z.Theta)
	// v = -Rzᵀ J h.t, J h.t = [-h.y, h.x]
	vx := cz*h.Y - sz*h.X
	vy := -sz*h.Y - cz*h.X

	jb = Mat3{
		chx, -shy, 0,
		shy, chx, 0,
		0, 0, 1,
	}
	ja = Mat3{
		-chx, shy, vx,
		-shy, -chx, vy,
		0, 0, -1,
	}
	return ja, jb
}

// priorJacobianWorld returns d(e)/d(p) for e = z⁻¹ ⊕ p, which is R(z)ᵀ
// acting on world-frame translation perturbations.
func priorJacobianWorld(z Pose2) Mat3 {
	cz, sz := math.Cos(z.Theta), math.Sin(z.Theta)
	return Mat3{
		cz, sz, 0,
		-sz, cz, 0,
		0, 0, 1,
	}
}

// ---------------------------------------------------------------------------
// Small 3x3 matrix helpers (row major).
// ---------------------------------------------------------------------------

func matIdentity() Mat3 {
	return Mat3{1, 0, 0, 0, 1, 0, 0, 0, 1}
}

func matScale(m Mat3, s float64) Mat3 {
	for i := 0; i < 9; i++ {
		m[i] *= s
	}
	return m
}

func matVec3(m Mat3, v Vec3) Vec3 {
	return Vec3{
		m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
		m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
		m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
	}
}

func matMul3(a, b Mat3) Mat3 {
	var r Mat3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			s := 0.0
			for k := 0; k < 3; k++ {
				s += a[i*3+k] * b[k*3+j]
			}
			r[i*3+j] = s
		}
	}
	return r
}

func transpose3(m Mat3) Mat3 {
	return Mat3{
		m[0], m[3], m[6],
		m[1], m[4], m[7],
		m[2], m[5], m[8],
	}
}

// symmetrize3 averages a matrix with its transpose to remove asymmetric noise.
func symmetrize3(m Mat3) Mat3 {
	t := transpose3(m)
	for i := 0; i < 9; i++ {
		m[i] = (m[i] + t[i]) * 0.5
	}
	return m
}

// invertMat3 inverts a generic 3x3 matrix via cofactors.
func invertMat3(m Mat3) (Mat3, bool) {
	const eps = 1e-18
	c := func(r, col int) float64 { return m[r*3+col] }
	det := c(0, 0)*(c(1, 1)*c(2, 2)-c(1, 2)*c(2, 1)) -
		c(0, 1)*(c(1, 0)*c(2, 2)-c(1, 2)*c(2, 0)) +
		c(0, 2)*(c(1, 0)*c(2, 1)-c(1, 1)*c(2, 0))
	if math.Abs(det) < eps || math.IsNaN(det) || math.IsInf(det, 0) {
		return Mat3{}, false
	}
	inv := det
	var r Mat3
	r[0] = (c(1, 1)*c(2, 2) - c(1, 2)*c(2, 1)) / inv
	r[1] = (c(0, 2)*c(2, 1) - c(0, 1)*c(2, 2)) / inv
	r[2] = (c(0, 1)*c(1, 2) - c(0, 2)*c(1, 1)) / inv
	r[3] = (c(1, 2)*c(2, 0) - c(1, 0)*c(2, 2)) / inv
	r[4] = (c(0, 0)*c(2, 2) - c(0, 2)*c(2, 0)) / inv
	r[5] = (c(0, 2)*c(1, 0) - c(0, 0)*c(1, 2)) / inv
	r[6] = (c(1, 0)*c(2, 1) - c(1, 1)*c(2, 0)) / inv
	r[7] = (c(0, 1)*c(2, 0) - c(0, 0)*c(2, 1)) / inv
	r[8] = (c(0, 0)*c(1, 1) - c(0, 1)*c(1, 0)) / inv
	return r, true
}

// cholSqrt3 returns an upper-triangular R with RᵀR = m for symmetric positive
// definite m (3x3 Cholesky, R = Lᵀ).
func cholSqrt3(m Mat3) (Mat3, bool) {
	var l Mat3
	for j := 0; j < 3; j++ {
		for i := j; i < 3; i++ {
			sum := m[i*3+j]
			for k := 0; k < j; k++ {
				sum -= l[i*3+k] * l[j*3+k]
			}
			if i == j {
				if sum <= 1e-18 || math.IsNaN(sum) || math.IsInf(sum, 0) {
					return Mat3{}, false
				}
				l[i*3+j] = math.Sqrt(sum)
			} else {
				l[i*3+j] = sum / l[j*3+j]
			}
		}
	}
	return transpose3(l), true
}

// covarianceInformation turns a measurement covariance into a symmetric
// positive definite information matrix by adding the configured diagonal
// floor, symmetrizing and inverting. The floor guarantees that zero or
// rank-deficient uncertainties never silently become infinite information.
func covarianceInformation(cov Mat3, floor Vec3) (Mat3, bool) {
	c := symmetrize3(cov)
	c[0] += floor[0]
	c[4] += floor[1]
	c[8] += floor[2]
	return invertMat3(c)
}
