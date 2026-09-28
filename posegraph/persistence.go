package posegraph

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
)

// snapshotPersister atomically writes JSON snapshots (write temp + fsync +
// rename) and reads the latest snapshot on startup. The rename step guarantees
// that a crash never leaves a half-written snapshot.
type snapshotPersister struct {
	path string
}

func newSnapshotPersister(path string) *snapshotPersister {
	return &snapshotPersister{path: path}
}

// save atomically persists the snapshot.
func (p *snapshotPersister) save(snap Snapshot) error {
	if p.path == "" {
		return errors.New("posegraph: snapshot path is empty")
	}
	data, err := encodeSnapshot(snap)
	if err != nil {
		return err
	}
	dir := filepath.Dir(p.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".posegraph-snap-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, p.path)
}

// load reads the snapshot. ok=false (with no error) means none exists yet.
func (p *snapshotPersister) load() (Snapshot, bool, error) {
	data, err := os.ReadFile(p.path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return Snapshot{}, false, nil
		}
		return Snapshot{}, false, err
	}
	var snap Snapshot
	if err := decodeSnapshot(data, &snap); err != nil {
		return Snapshot{}, false, err
	}
	return snap, true, nil
}

// decodeSnapshot parses a snapshot previously produced by encodeSnapshot.
func decodeSnapshot(data []byte, snap *Snapshot) error {
	return jsonUnmarshal(data, snap)
}
