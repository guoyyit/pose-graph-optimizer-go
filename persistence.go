package posegraph

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Persistence format (all files inside Config.PersistenceDir):
//
//	snapshot.json  - last full consistent snapshot (graph + version)
//	wal.log        - append-only JSON-line mutation records since snapshot
//
// The snapshot is written atomically (temp file + fsync + rename); each WAL
// line is fsynced before the mutation is published, so a crash loses at most
// a mutation whose caller already saw an I/O error.

type recordKind int

const (
	recNode recordKind = iota + 1
	recConstraint
)

type walRecord struct {
	Kind       recordKind  `json:"kind"`
	Node       *Node       `json:"node,omitempty"`
	Constraint *Constraint `json:"constraint,omitempty"`
}

// edgeJSON is the persisted form of Edge including internal state.
type edgeJSON struct {
	Constraint
	State     EdgeState `json:"state"`
	Weight    float64   `json:"weight"`
	Reason    string    `json:"reason"`
	FusedInto string    `json:"fused_into,omitempty"`
}

type snapshot struct {
	Version      int64      `json:"version"`
	Nodes        []Node     `json:"nodes"`
	Edges        []edgeJSON `json:"edges"`
	Pending      []edgeJSON `json:"pending"`
	NextSeq      int        `json:"next_seq"`
	GateRejected []gateJSON `json:"gate_rejected"`
	Mutations    int        `json:"mutations_since_snapshot"`
}

type gateJSON struct {
	EdgeID string    `json:"edge_id"`
	Event  GateEvent `json:"event"`
}

// store abstracts persistence so tests can swap in a fake.
type store interface {
	loadSnapshot() (*snapshot, []walRecord, error)
	writeSnapshot(*snapshot) error
	appendRecord(rec walRecord) error
	io.Closer
}

// fileStore is the directory-backed implementation.
type fileStore struct {
	dir           string
	wal           *os.File
	snapshotEvery int
}

func newFileStore(dir string, snapshotEvery int) (store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("posegraph: create persistence dir: %w", err)
	}
	f, err := os.OpenFile(filepath.Join(dir, "wal.log"),
		os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &fileStore{dir: dir, wal: f, snapshotEvery: snapshotEvery}, nil
}

func (fs *fileStore) snapshotPath() string { return filepath.Join(fs.dir, "snapshot.json") }
func (fs *fileStore) walPath() string      { return filepath.Join(fs.dir, "wal.log") }

func (fs *fileStore) loadSnapshot() (*snapshot, []walRecord, error) {
	snap := &snapshot{}
	if b, err := os.ReadFile(fs.snapshotPath()); err == nil {
		if err := json.Unmarshal(b, snap); err != nil {
			return nil, nil, fmt.Errorf("posegraph: corrupt snapshot: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return nil, nil, err
	}
	recs, err := readWAL(fs.walPath())
	if err != nil {
		return nil, nil, err
	}
	return snap, recs, nil
}

func readWAL(path string) ([]walRecord, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var recs []walRecord
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var r walRecord
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return nil, fmt.Errorf("posegraph: corrupt wal: %w", err)
		}
		recs = append(recs, r)
	}
	return recs, nil
}

func (fs *fileStore) writeSnapshot(s *snapshot) error {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	tmp := fs.snapshotPath() + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(b); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, fs.snapshotPath()); err != nil {
		return err
	}
	// Snapshot now supersedes the WAL.
	if err := fs.wal.Close(); err != nil {
		return err
	}
	if err := os.WriteFile(fs.walPath(), nil, 0o644); err != nil {
		return err
	}
	nf, err := os.OpenFile(fs.walPath(), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if err := nf.Sync(); err != nil {
		nf.Close()
		return err
	}
	fs.wal = nf
	return nil
}

func (fs *fileStore) appendRecord(r walRecord) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if _, err := fs.wal.Write(b); err != nil {
		return err
	}
	return fs.wal.Sync()
}

func (fs *fileStore) Close() error {
	if fs.wal != nil {
		return fs.wal.Close()
	}
	return nil
}

// ---- service-side persistence glue ----

func (s *Service) buildSnapshotLocked() *snapshot {
	snap := &snapshot{Version: s.version, NextSeq: s.graph.nextSeq, Mutations: s.mutationsSinceSnapshot}
	for _, id := range s.graph.NodeIDs() {
		nd, _ := s.graph.Node(id)
		snap.Nodes = append(snap.Nodes, nd)
	}
	for _, id := range sortedStringKeys(s.graph.edges) {
		e := s.graph.edges[id]
		snap.Edges = append(snap.Edges, edgeJSON{
			Constraint: e.Constraint, State: e.state, Weight: e.weight,
			Reason: e.reason, FusedInto: e.fusedInto,
		})
	}
	for _, id := range sortedStringKeys(s.graph.pending) {
		e := s.graph.pending[id]
		snap.Pending = append(snap.Pending, edgeJSON{
			Constraint: e.Constraint, State: e.state, Weight: e.weight,
			Reason: e.reason, FusedInto: e.fusedInto,
		})
	}
	ids := make([]string, 0, len(s.gateRejected))
	for id := range s.gateRejected {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		ev := s.gateRejected[id]
		snap.GateRejected = append(snap.GateRejected, gateJSON{EdgeID: id, Event: ev})
	}
	return snap
}

func sortedStringKeys[V any](m map[string]*V) []string {
	ids := make([]string, 0, len(m))
	for id := range m {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func (s *Service) writeSnapshotLocked() error {
	if s.store == nil {
		return nil
	}
	return s.store.writeSnapshot(s.buildSnapshotLocked())
}

func (s *Service) maybeSnapshotLocked() error {
	every := s.cfg.SnapshotEvery
	if every <= 0 {
		every = 100
	}
	if s.mutationsSinceSnapshot < every {
		return nil
	}
	s.mutationsSinceSnapshot = 0
	return s.writeSnapshotLocked()
}

// restore rebuilds the in-memory graph from the last snapshot and replays
// the WAL deterministically, then re-optimizes. Accepted constraints cannot
// be lost: every published mutation is in either snapshot or WAL.
func (s *Service) restore() error {
	snap, recs, err := s.store.loadSnapshot()
	if err != nil {
		return err
	}
	if snap == nil {
		snap = &snapshot{}
	}

	g := NewGraph()
	g.dupChi2Threshold = s.cfg.DuplicateChi2Threshold
	g.nextSeq = snap.NextSeq

	// Nodes first in snapshot order, then replay.
	for _, n := range snap.Nodes {
		g.AddNode(n)
	}
	// Restore edges in snapshot order (sorted for determinism).
	restored := append([]edgeJSON(nil), snap.Edges...)
	pending := append([]edgeJSON(nil), snap.Pending...)
	sort.Slice(restored, func(i, j int) bool { return restored[i].ID < restored[j].ID })
	for _, ej := range restored {
		e := ej
		// Route through the normal merge path so rules stay identical, but
		// preserve recorded rejection/fusion verdicts afterwards.
		id, err := g.AddConstraint(e.Constraint)
		if err == ErrPending {
			g.pending[id] = edgeFromJSON(e)
		}
		if ge, ok := g.edges[id]; ok {
			ge.state = e.State
			ge.weight = e.Weight
			ge.reason = e.Reason
			ge.fusedInto = e.FusedInto
		}
	}
	for _, ej := range pending {
		g.pending[ej.Constraint.ID] = edgeFromJSON(ej)
	}
	for _, gj := range snap.GateRejected {
		s.gateRejected[gj.EdgeID] = gj.Event
	}
	s.mutationsSinceSnapshot = snap.Mutations
	s.graph = g
	s.version = snap.Version

	// Replay WAL records (they occurred after the snapshot).
	for _, r := range recs {
		switch r.Kind {
		case recNode:
			if r.Node != nil {
				g.AddNode(*r.Node)
			}
		case recConstraint:
			if r.Constraint != nil {
				if _, err := g.AddConstraint(*r.Constraint); err != nil && err != ErrPending {
					return err
				}
			}
		}
	}
	return nil
}

func edgeFromJSON(e edgeJSON) *Edge {
	return &Edge{
		Constraint: e.Constraint, state: e.State, weight: e.Weight,
		reason: e.Reason, fusedInto: e.FusedInto,
	}
}
