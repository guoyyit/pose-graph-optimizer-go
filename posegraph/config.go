package posegraph

// defaultConfig returns conservative defaults tuned for 2D pose graphs with
// coordinates in metres and angles in radians.
func defaultConfig() Config {
	return Config{
		MaxIterations:     50,
		ConvergeChi2:      1e-7,
		ConvergeStep:      1e-8,
		StagnationRounds:  5,
		InitialDamping:    1e-3,
		DampingUp:         10,
		DampingDown:       0.5,
		MaxDamping:        1e12,
		ConditionLimit:    1e12,
		CovarianceFloor:   Vec3{1e-6, 1e-6, 1e-4},
		LoopChi2Gate:      7.815, // chi-square 3 DOF at 95%
		RobustKernel:      1.0,
		ConflictInflation: 4.0,
		AutoOptimize:      true,
		SnapshotPath:      "",
	}
}

// normalize fills unset/zero fields with defaults and clamps unsafe values.
// It is called on every store creation and after loading a snapshot, which
// also pins behavior when old snapshots lack newer config fields.
func (c *Config) normalize() {
	d := defaultConfig()
	if c.MaxIterations <= 0 {
		c.MaxIterations = d.MaxIterations
	}
	if c.ConvergeChi2 <= 0 {
		c.ConvergeChi2 = d.ConvergeChi2
	}
	if c.ConvergeStep <= 0 {
		c.ConvergeStep = d.ConvergeStep
	}
	if c.StagnationRounds <= 0 {
		c.StagnationRounds = d.StagnationRounds
	}
	if c.InitialDamping <= 0 {
		c.InitialDamping = d.InitialDamping
	}
	if c.DampingUp <= 1 {
		c.DampingUp = d.DampingUp
	}
	if c.DampingDown <= 0 || c.DampingDown >= 1 {
		c.DampingDown = d.DampingDown
	}
	if c.MaxDamping <= 0 {
		c.MaxDamping = d.MaxDamping
	}
	if c.ConditionLimit <= 0 {
		c.ConditionLimit = d.ConditionLimit
	}
	if c.CovarianceFloor[0] < 0 || c.CovarianceFloor[1] < 0 || c.CovarianceFloor[2] < 0 {
		c.CovarianceFloor = d.CovarianceFloor
	} else if c.CovarianceFloor == (Vec3{}) {
		c.CovarianceFloor = d.CovarianceFloor
	}
	if c.LoopChi2Gate <= 0 {
		c.LoopChi2Gate = d.LoopChi2Gate
	}
	if c.RobustKernel < 0 {
		c.RobustKernel = d.RobustKernel
	}
	if c.ConflictInflation < 1 {
		c.ConflictInflation = d.ConflictInflation
	}
}
