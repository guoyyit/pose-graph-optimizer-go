package posegraph

import "math"

func sqrt(x float64) float64 { return math.Sqrt(max(0, x)) }
func abs(x float64) float64  { return math.Abs(x) }
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
