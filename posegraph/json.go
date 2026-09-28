package posegraph

import "encoding/json"

func jsonUnmarshal(data []byte, v any) error { return json.Unmarshal(data, v) }

// encodeSnapshot is the deterministic JSON encoder used on disk and by callers
// that wish to inspect the serialized form.
func encodeSnapshot(snap Snapshot) ([]byte, error) {
	return json.MarshalIndent(snap, "", "  ")
}
