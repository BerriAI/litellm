package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

// Workload describes an open-loop test: requests arrive as a Poisson process at the stage's
// rate, each one drawn from a weighted class.
type Workload struct {
	Name   string  `json:"name"`
	Stages []Stage `json:"stages"`
	// WarmupS seconds at the first stage's rate whose requests are marked warmup.
	WarmupS float64 `json:"warmup_s"`
	// MaxInFlight caps open requests per load generator; arrivals above it are counted as shed.
	MaxInFlight int `json:"max_in_flight"`
	// NewConnFraction of requests use a fresh TCP connection, like many separate clients.
	NewConnFraction float64           `json:"new_conn_fraction"`
	Keys            KeySpec           `json:"keys"`
	Models          map[string]Models `json:"models"`
	Classes         []Class           `json:"classes"`
}

type Stage struct {
	DurationS float64 `json:"duration_s"`
	RPS       float64 `json:"rps"`
}

type KeySpec struct {
	File string `json:"file"`
	// ZipfS > 1 skews traffic towards a few hot keys; 0 picks keys uniformly.
	ZipfS float64 `json:"zipf_s"`
}

// Models names the model groups for one endpoint: Prefix + 0..Count-1, matching genconfig.
type Models struct {
	Prefix string `json:"prefix"`
	Count  int    `json:"count"`
}

type Class struct {
	Name           string  `json:"name"`
	Weight         float64 `json:"weight"`
	Endpoint       string  `json:"endpoint"` // "chat" or "messages"
	StreamFraction float64 `json:"stream_fraction"`
	ContextTokens  Dist    `json:"context_tokens"`
	TTFTMs         Dist    `json:"ttft_ms"`
	DurationS      Dist    `json:"duration_s"`
	// OutputTokensPerS sets output length from the duration (default 50).
	OutputTokensPerS float64 `json:"output_tokens_per_s"`
}

// Dist is a distribution: "const" (Min), "uniform", "loguniform" (Min..Max), or "choice"
// (Values, optional Weights).
type Dist struct {
	Dist    string    `json:"dist"`
	Min     float64   `json:"min"`
	Max     float64   `json:"max"`
	Values  []float64 `json:"values"`
	Weights []float64 `json:"weights"`
}

func (d Dist) Sample(r *rand.Rand) float64 {
	switch d.Dist {
	case "", "const":
		return d.Min
	case "uniform":
		return d.Min + r.Float64()*(d.Max-d.Min)
	case "loguniform":
		lo, hi := math.Log(d.Min), math.Log(d.Max)
		return math.Exp(lo + r.Float64()*(hi-lo))
	case "choice":
		return d.Values[weighted(r, d.Weights, len(d.Values))]
	}
	panic("unknown dist " + d.Dist)
}

func (d Dist) Upper() float64 {
	switch d.Dist {
	case "", "const":
		return d.Min
	case "choice":
		m := 0.0
		for _, v := range d.Values {
			m = max(m, v)
		}
		return m
	}
	return d.Max
}

func weighted(r *rand.Rand, weights []float64, n int) int {
	if len(weights) != n {
		return r.IntN(n)
	}
	total := 0.0
	for _, w := range weights {
		total += w
	}
	x := r.Float64() * total
	for i, w := range weights {
		if x < w {
			return i
		}
		x -= w
	}
	return n - 1
}

func loadWorkload(path string) (*Workload, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var w Workload
	if err := json.Unmarshal(b, &w); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(w.Stages) == 0 || len(w.Classes) == 0 {
		return nil, fmt.Errorf("%s: needs stages and classes", path)
	}
	for _, c := range w.Classes {
		if c.Endpoint != "chat" && c.Endpoint != "messages" {
			return nil, fmt.Errorf("class %s: endpoint must be chat or messages", c.Name)
		}
		if m, ok := w.Models[c.Endpoint]; !ok || m.Count == 0 {
			return nil, fmt.Errorf("class %s: no models for endpoint %s", c.Name, c.Endpoint)
		}
	}
	if w.MaxInFlight == 0 {
		w.MaxInFlight = 100000
	}
	return &w, nil
}

// longest is the longest a request of this workload can take, for the final drain.
func (w *Workload) longest() float64 {
	m := 0.0
	for _, c := range w.Classes {
		m = max(m, c.DurationS.Upper())
	}
	return m
}
