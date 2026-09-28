package posegraph

import "encoding/json"

func encodeAny(v any) ([]byte, error) { return json.Marshal(v) }
