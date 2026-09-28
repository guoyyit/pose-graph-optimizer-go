package posegraph

import "math"

// Vec3 is a 3-vector (dx, dy, dtheta) in the SE(2) tangent space.
type Vec3 [3]float64

// Mat3 is a row-major 3x3 matrix.
type Mat3 [9]float64

// Pose is an SE(2) pose: translation (X,Y) and heading Theta in radians.
type Pose struct {
	X     float64
	Y     float64
	Theta float64
}

// WrapAngle normalizes an angle to [-pi, pi).
func WrapAngle(a float64) float64 {
	for a >= math.Pi {
		a -= 2 * math.Pi
	}
	for a < -math.Pi {
		a += 2 * math.Pi
	}
	return a
}

// Compose returns the group product a*b: apply b in the frame of a.
//
//	x   = cos(ta) xb - sin(ta) yb + xa
//	y   = sin(ta) xb + cos(ta) yb + ya
//	th  = wrap(ta + tb)
func Compose(a, b Pose) Pose {
	c, s := math.Cos(a.Theta), math.Sin(a.Theta)
	return Pose{
		X:     c*b.X - s*b.Y + a.X,
		Y:     s*b.X + c*b.Y + a.Y,
		Theta: WrapAngle(a.Theta + b.Theta),
	}
}

// Inverse returns the group inverse of p.
func Inverse(p Pose) Pose {
	c, s := math.Cos(p.Theta), math.Sin(p.Theta)
	return Pose{
		X:     -c*p.X - s*p.Y,
		Y:     s*p.X - c*p.Y,
		Theta: WrapAngle(-p.Theta),
	}
}

// Exp maps an SE(2) tangent vector v to a pose, using the closed-form
// exponential: R(theta) with translation V(theta)*(vx,vy).
func Exp(v Vec3) Pose {
	th := v[2]
	if math.Abs(th) < 1e-10 {
		// Second-order Taylor of V for numerical stability near zero.
		t2 := th * th
		ax := v[0] - th*v[1]/2 - t2*v[0]/6
		ay := th*v[0]/2 + v[1] - t2*v[1]/6
		return Pose{X: ax, Y: ay, Theta: WrapAngle(th)}
	}
	c, s := math.Cos(th), math.Sin(th)
	// V = 1/th * [[sin th, cos th - 1], [1 - cos th, sin th]]
	vs := s / th
	vc := (1 - c) / th
	x := vs*v[0] - vc*v[1]
	y := vc*v[0] + vs*v[1]
	return Pose{X: x, Y: y, Theta: WrapAngle(th)}
}

// Log maps an SE(2) pose to its tangent vector, the inverse of Exp.
func Log(p Pose) Vec3 {
	th := WrapAngle(p.Theta)
	if math.Abs(th) < 1e-10 {
		t2 := th * th
		// Invert the second-order V via a matching Taylor of V^-1.
		// V^-1 ~ [[1 - th^2/12, th/2],[-th/2, 1 - th^2/12]]
		a := 1 - t2/12
		b := th / 2
		return Vec3{
			a*p.X + b*p.Y,
			-b*p.X + a*p.Y,
			th,
		}
	}
	c, s := math.Cos(th), math.Sin(th)
	// V^-1 = th/(2(1-c)) [[s, 1-c],[c-1, s]]
	den := 2 * (1 - c)
	k := th / den
	vx := k * (s*p.X + (1-c)*p.Y)
	vy := k * ((c-1)*p.X + s*p.Y)
	return Vec3{vx, vy, th}
}

// AdjSE2 returns the adjoint action Ad_T of pose T on a tangent vector:
// a perturbation expressed in the body frame is re-expressed in the world
// frame as Ad_T * xi.
//
//	Ad = [[R, J t],[0, 1]] with J = [[0,-1],[1,0]]
func AdjSE2(t Pose) Mat3 {
	c, s := math.Cos(t.Theta), math.Sin(t.Theta)
	return Mat3{
		c, -s, t.Y,
		s, c, -t.X,
		0, 0, 1,
	}
}
