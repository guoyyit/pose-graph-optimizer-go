// Package posegraph implements a local 2D pose-graph optimizer with loop
// closure fusion, incremental maintenance, diagnostics, versioned concurrent
// access and snapshot persistence.
package posegraph

import "time"

// NodeID identifies a pose node in the graph.
type NodeID string

// ConstraintKind discriminates between relative-pose odometry/loop constraints
// and absolute prior constraints that anchor a node to a global frame.
type ConstraintKind int

const (
	// Relative is a relative pose constraint z between two nodes.
	Relative ConstraintKind = iota
	// Prior is an absolute pose constraint that anchors one node.
	Prior
)

// Pose2 is a 2D rigid-body pose: (x, y) translation and theta rotation in
// radians, normalized to (-pi, pi].
type Pose2 struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Theta float64 `json:"theta"`
}

// Mat3 is a row-major 3x3 matrix (typically a covariance or information).
type Mat3 [9]float64

// Vec3 is a 3-vector (x, y, theta).
type Vec3 [3]float64

// Node is a pose vertex. Initial is the pose reported when the node was first
// submitted; Fixed nodes are not moved by the optimizer.
type Node struct {
	ID         NodeID `json:"id"`
	Initial    Pose2  `json:"initial"`
	Fixed      bool   `json:"fixed"`
	FirstAdded int64  `json:"first_added"`
}

// Constraint is a measurement edge.
//
// For Relative constraints it states: pose(B) = pose(A) (+) Measurement.
// For Prior constraints A must be empty and Measurement is the absolute pose.
//
// Covariance is the measurement covariance in the (x, y, theta) frame. A
// non-zero Information matrix is used directly; otherwise Information is
// derived from Covariance (which, after the diagonal floor, must be invertible).
type Constraint struct {
	ID          string         `json:"id"`
	Kind        ConstraintKind `json:"kind"`
	A           NodeID         `json:"a,omitempty"`
	B           NodeID         `json:"b,omitempty"`
	Measurement Pose2          `json:"measurement"`
	Covariance  Mat3           `json:"covariance"`
	Information Mat3           `json:"information"`
	// Loop marks a relative constraint as a loop-closure candidate that must
	// pass the consistency gate before being accepted at full weight.
	Loop bool `json:"loop"`
	// RobustWeight is the current robust-kernel multiplier applied to the
	// information matrix (1 means full weight).
	RobustWeight float64 `json:"robust_weight"`
	Accepted     bool    `json:"accepted"`
	// Rejected records why a loop constraint was rejected; empty otherwise.
	Rejected   string `json:"rejected,omitempty"`
	FirstAdded int64  `json:"first_added"`
}

// Residual is one constraint's 3-dimensional normalized residual vector
// (whitened: sqrtInfo * rawError), its L2 norm and per-component raw error.
type Residual struct {
	ConstraintID string  `json:"constraint_id"`
	Raw          Vec3    `json:"raw"`
	Whitened     Vec3    `json:"whitened"`
	Norm2        float64 `json:"norm2"`
}

// NodeResidual aggregates the information-weighted residual contribution at a
// node and the largest whitened norm among incident constraints.
type NodeResidual struct {
	NodeID      NodeID  `json:"node_id"`
	Force       Vec3    `json:"force"`
	MaxWhitened float64 `json:"max_whitened"`
	Degree      int     `json:"degree"`
}

// Marginal is a node's estimated global pose and its marginal covariance.
type Marginal struct {
	ID         NodeID `json:"id"`
	Pose       Pose2  `json:"pose"`
	Covariance Mat3   `json:"covariance"`
	Fixed      bool   `json:"fixed"`
}

// Status classifies the outcome of an optimization run.
type Status int

const (
	// StatusNotRun means optimization has never successfully completed.
	StatusNotRun Status = iota
	// StatusConverged means a solution meeting tolerances was found.
	StatusConverged
	// StatusMaxIterations means progress was being made at the iteration cap.
	StatusMaxIterations
	// StatusStagnated means the optimizer stopped making meaningful progress
	// before meeting tolerances.
	StatusStagnated
	// StatusIllConditioned means the normal-equations matrix could not be
	// factored reliably.
	StatusIllConditioned
	// StatusFailed means the solve diverged or encountered a numeric failure.
	StatusFailed
)

// String renders a Status.
func (s Status) String() string {
	switch s {
	case StatusConverged:
		return "converged"
	case StatusMaxIterations:
		return "max_iterations"
	case StatusStagnated:
		return "stagnated"
	case StatusIllConditioned:
		return "ill_conditioned"
	case StatusFailed:
		return "failed"
	default:
		return "not_run"
	}
}

// Diagnostics carries solver health information for the latest run.
type Diagnostics struct {
	Status         Status    `json:"status"`
	Iterations     int       `json:"iterations"`
	FinalChi2      float64   `json:"final_chi2"`
	MaxWhitened    float64   `json:"max_whitened"`
	RMSE           float64   `json:"rmse"`
	CostDelta      float64   `json:"cost_delta"`
	StepNorm       float64   `json:"step_norm"`
	ConditionEst   float64   `json:"condition_est"`
	Warnings       []string  `json:"warnings"`
	NumConstraints int       `json:"num_constraints"`
	NumNodes       int       `json:"num_nodes"`
	OptimizedAt    time.Time `json:"optimized_at"`
}

// Result is an immutable view of one consistent optimization version.
type Result struct {
	Version   int64                   `json:"version"`
	Nodes     map[NodeID]Marginal     `json:"nodes"`
	Residuals map[string]Residual     `json:"residuals"`
	PerNode   map[NodeID]NodeResidual `json:"per_node"`
	Diag      Diagnostics             `json:"diagnostics"`
	// EstimatedAt is the wall-clock time the version was produced.
	EstimatedAt time.Time `json:"estimated_at"`
}

// Config holds optimizer, gate and persistence parameters.
type Config struct {
	// MaxIterations bounds Gauss-Newton/LM iterations (default 50).
	MaxIterations int
	// ConvergeChi2 is the relative cost decrease considered convergence
	// (default 1e-7).
	ConvergeChi2 float64
	// ConvergeStep is the step-norm threshold for convergence (default 1e-8).
	ConvergeStep float64
	// StagnationRounds is the number of consecutive non-progressing LM rounds
	// before declaring stagnation (default 5).
	StagnationRounds int
	// InitialDamping is the starting LM damping factor (default 1e-3).
	InitialDamping float64
	// DampingUp / DampingDown scale damping on failed/successful steps
	// (defaults 10 and 0.5).
	DampingUp, DampingDown float64
	// MaxDamping aborts with a failure once damping exceeds it (default 1e12).
	MaxDamping float64
	// ConditionLimit rejects factorization above this estimated reciprocal
	// condition number threshold (default 1e12).
	ConditionLimit float64
	// CovarianceFloor is added to each measurement-covariance diagonal to
	// guard against zero uncertainties (default 1e-6 / 1e-4 for theta).
	CovarianceFloor Vec3
	// LoopChi2Gate is the per-DOF chi2 threshold above which a loop candidate
	// is rejected (default 7.815 ~ chi-square 95% with 3 DOF).
	LoopChi2Gate float64
	// RobustKernel is the DCS/Huber-style delta applied to accepted outliers;
	// 0 disables robust down-weighting (default 1.0 whitened norm).
	RobustKernel float64
	// ConflictInflation multiplies covariance when conflicting duplicates are
	// fused (default 4.0).
	ConflictInflation float64
	// AutoOptimize reruns the optimizer after every mutating commit when true.
	AutoOptimize bool
	// SnapshotPath enables atomic JSON persistence when non-empty.
	SnapshotPath string
}

// DefaultConfig returns a configuration with conservative defaults.
func DefaultConfig() Config { return defaultConfig() }

// ChangeKind classifies a submitted change during commit processing.
type ChangeKind int

const (
	// ChangeNode adds or repeats a node.
	ChangeNode ChangeKind = iota
	// ChangeConstraint adds, repeats or fuses a constraint.
	ChangeConstraint
)

// Change is one unit of incremental input.
type Change struct {
	Kind       ChangeKind
	Node       Node
	Constraint Constraint
}

// ChangeOutcome reports how an individual change was handled.
type ChangeOutcome string

const (
	OutcomeAdded          ChangeOutcome = "added"
	OutcomeRepeated       ChangeOutcome = "repeated_ignored"
	OutcomeFused          ChangeOutcome = "fused_conflict"
	OutcomePending        ChangeOutcome = "pending_dangling"
	OutcomeLoopAccept     ChangeOutcome = "loop_accepted"
	OutcomeLoopDownweight ChangeOutcome = "loop_downweighted"
	OutcomeRejected       ChangeOutcome = "rejected"
	OutcomeInvalid        ChangeOutcome = "invalid"
)

// ChangeReport is the per-change processing report.
type ChangeReport struct {
	Change     Change        `json:"-"`
	Outcome    ChangeOutcome `json:"outcome"`
	Detail     string        `json:"detail,omitempty"`
	Weight     float64       `json:"weight,omitempty"`
	Chi2PerDOF float64       `json:"chi2_per_dof,omitempty"`
}

// CommitReport is the deterministic result of applying one commit.
type CommitReport struct {
	VersionBefore int64          `json:"version_before"`
	VersionAfter  int64          `json:"version_after"`
	Reports       []ChangeReport `json:"reports"`
	Optimization  *Diagnostics   `json:"optimization,omitempty"`
}

// LoopDecision records the evaluation of a loop-closure candidate.
type LoopDecision struct {
	ConstraintID string  `json:"constraint_id"`
	Chi2PerDOF   float64 `json:"chi2_per_dof"`
	Weight       float64 `json:"weight"`
	Accepted     bool    `json:"accepted"`
	Downweighted bool    `json:"downweighted"`
	Reason       string  `json:"reason,omitempty"`
}

// Snapshot is the persisted, restart-safe representation of a Store.
type Snapshot struct {
	Version     int64                 `json:"version"`
	Nodes       map[NodeID]Node       `json:"nodes"`
	Constraints map[string]Constraint `json:"constraints"`
	Pending     []Constraint          `json:"pending"`
	Rejected    []Constraint          `json:"rejected"`
	NextSeq     int64                 `json:"next_seq"`
	LastResult  *Result               `json:"last_result,omitempty"`
	Config      Config                `json:"config"`
	SavedAt     time.Time             `json:"saved_at"`
}
