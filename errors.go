package posegraph

import "errors"

var notImplemented = errors.New("posegraph: not implemented")

// Sentinel errors returned by public APIs.
var (
	// ErrConstraintRejected indicates a constraint failed the consistency
	// gate and was not incorporated into the estimate.
	ErrConstraintRejected = errors.New("posegraph: constraint rejected by consistency gate")
	// ErrDegenerate indicates the normal equations are (numerically) singular.
	ErrDegenerate = errors.New("posegraph: degenerate or ill-conditioned system")
	// ErrNotConverged indicates the optimizer stalled before convergence.
	ErrNotConverged = errors.New("posegraph: optimizer did not converge")
	// ErrNoNodes indicates an optimization was requested on an empty graph.
	ErrNoNodes = errors.New("posegraph: graph has no nodes")
	// ErrInvalidConstraint indicates a malformed constraint (bad covariance etc).
	ErrInvalidConstraint = errors.New("posegraph: invalid constraint")
	// ErrPending indicates a constraint references nodes that do not exist yet.
	ErrPending = errors.New("posegraph: constraint pending missing endpoint")
)
