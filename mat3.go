package posegraph

// Small fixed-size 3x3 linear algebra used for covariances and information.

func vecAdd(a, b Vec3) Vec3 { return Vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }
func vecSub(a, b Vec3) Vec3 { return Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }
func vecScale(a Vec3, s float64) Vec3 {
	return Vec3{a[0] * s, a[1] * s, a[2] * s}
}
func vecDot(a, b Vec3) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
func vecNorm(a Vec3) float64 {
	return sqrt(vecDot(a, a))
}

func matVec(m Mat3, v Vec3) Vec3 {
	return Vec3{
		m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
		m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
		m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
	}
}

func matMul(a, b Mat3) Mat3 {
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

func matTranspose(m Mat3) Mat3 {
	return Mat3{
		m[0], m[3], m[6],
		m[1], m[4], m[7],
		m[2], m[5], m[8],
	}
}

func matDet(m Mat3) float64 {
	return m[0]*(m[4]*m[8]-m[5]*m[7]) -
		m[1]*(m[3]*m[8]-m[5]*m[6]) +
		m[2]*(m[3]*m[7]-m[4]*m[6])
}

// matInverse inverts a general 3x3 matrix; ok=false if singular.
func matInverse(m Mat3) (Mat3, bool) {
	d := matDet(m)
	if abs(d) < 1e-15 {
		return Mat3{}, false
	}
	inv := 1 / d
	var r Mat3
	r[0] = (m[4]*m[8] - m[5]*m[7]) * inv
	r[1] = (m[2]*m[7] - m[1]*m[8]) * inv
	r[2] = (m[1]*m[5] - m[2]*m[4]) * inv
	r[3] = (m[5]*m[6] - m[3]*m[8]) * inv
	r[4] = (m[0]*m[8] - m[2]*m[6]) * inv
	r[5] = (m[2]*m[3] - m[0]*m[5]) * inv
	r[6] = (m[3]*m[7] - m[4]*m[6]) * inv
	r[7] = (m[1]*m[6] - m[0]*m[7]) * inv
	r[8] = (m[0]*m[4] - m[1]*m[3]) * inv
	return r, true
}

// symInverse inverts a symmetric 3x3 matrix (using symmetrized input to be
// safe against tiny asymmetries from accumulation).
func symInverse(m Mat3) (Mat3, bool) {
	s := Mat3{
		m[0],
		(m[1] + m[3]) / 2,
		(m[2] + m[6]) / 2,
		(m[1] + m[3]) / 2,
		m[4],
		(m[5] + m[7]) / 2,
		(m[2] + m[6]) / 2,
		(m[5] + m[7]) / 2,
		m[8],
	}
	r, ok := matInverse(s)
	if !ok {
		return Mat3{}, false
	}
	// Enforce symmetry of the result.
	r[1] = (r[1] + r[3]) / 2
	r[3] = r[1]
	r[2] = (r[2] + r[6]) / 2
	r[6] = r[2]
	r[5] = (r[5] + r[7]) / 2
	r[7] = r[5]
	return r, true
}

func identity3() Mat3 {
	return Mat3{1, 0, 0, 0, 1, 0, 0, 0, 1}
}

func zero3() Mat3 { return Mat3{} }

// symmetricPart returns (m + m^T)/2.
func symmetricPart(m Mat3) Mat3 {
	return Mat3{
		m[0], (m[1] + m[3]) / 2, (m[2] + m[6]) / 2,
		(m[1] + m[3]) / 2, m[4], (m[5] + m[7]) / 2,
		(m[2] + m[6]) / 2, (m[5] + m[7]) / 2, m[8],
	}
}
