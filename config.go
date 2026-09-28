package posegraph

// Config controls optimization, robust gating and persistence behavior.
// The zero value is not valid; use DefaultConfig and override fields.
type Config struct {
	// AnchorID fixes the gauge freedom by holding this node at its initial
	// pose. If empty, the lexicographically smallest node ID is anchored.
	AnchorID string

	// MaxIterations bounds the Levenberg-Marquardt iterations.
	MaxIterations int
	// FunctionTolerance stops when the relative cost decrease falls below it.
	FunctionTolerance float64
	// StepTolerance stops when the maximum tangent step falls below it.
	StepTolerance float64
	// GradientTolerance stops when the inf-norm of the gradient falls below it.
	GradientTolerance float64

	// InitialDamping is the starting LM damping lambda.
	InitialDamping float64
	// DampingGrowth / DampingShrink adjust lambda on rejected/accepted steps.
	DampingGrowth float64
	DampingShrink float64
	// MinPivotRel is the relative Cholesky pivot tolerance; below it the
	// system is declared ill-conditioned.
	MinPivotRel float64
	// MaxCondition rejects solutions whose estimated condition number exceeds it.
	MaxCondition float64

	// GateConfidence is the chi-square confidence (e.g. 0.99) for rejecting
	// grossly inconsistent loop-closure constraints (DoF = 3).
	GateConfidence float64
	// DownweightConfidence is the lower band: chi2 in
	// [chi2(DownweightConfidence), chi2(GateConfidence)] is soft-downweighted.
	DownweightConfidence float64
	// MaxResidualChi2 is an absolute sanity ceiling: a "converged" solution
	// whose worst weighted per-constraint chi2 stays above it is reported as
	// a failure rather than a misleading success.
	MaxResidualChi2 float64
	// HuberDelta (in standardized residual units) applies a Huber kernel to
	// loop closures; <= 0 disables robust weighting.
	HuberDelta float64

	// DuplicateChi2Threshold fuses duplicate constraints whose measurements
	// agree within this chi2; otherwise the pair is recorded as a conflict
	// and the strongest (largest information) one wins.
	DuplicateChi2Threshold float64

	// PersistenceDir, when non-empty, enables snapshot + WAL persistence.
	PersistenceDir string
	// SnapshotEvery forces a fresh snapshot after this many accepted mutations.
	SnapshotEvery int
}

// DefaultConfig returns a sane default configuration.
func DefaultConfig() Config {
	return Config{
		MaxIterations:          50,
		FunctionTolerance:      1e-9,
		StepTolerance:          1e-8,
		GradientTolerance:      1e-10,
		InitialDamping:         1e-4,
		DampingGrowth:          10,
		DampingShrink:          0.5,
		MinPivotRel:            1e-10,
		MaxCondition:           1e14,
		GateConfidence:         0.99,
		DownweightConfidence:   0.95,
		MaxResidualChi2:        25, // 5 sigma standardized per edge
		HuberDelta:             4.0,
		DuplicateChi2Threshold: 7.815, // 95% chi2 dof=3
		SnapshotEvery:          100,
	}
}
