// summarize merges loadgen result files (one per generator) and prints markdown tables of proxy
// overhead: observed time minus the time mockllm was told to take.
//
//	summarize results/*.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
)

type result struct {
	T          int64   `json:"t"`
	Stage      int     `json:"stage"`
	Warmup     bool    `json:"warmup"`
	Class      string  `json:"class"`
	Endpoint   string  `json:"ep"`
	Stream     bool    `json:"stream"`
	CtxTokens  int     `json:"ctx"`
	OutTokens  int     `json:"out"`
	TTFTPlanMs float64 `json:"ttft_plan_ms"`
	DurPlanMs  float64 `json:"dur_plan_ms"`
	Status     int     `json:"status"`
	OK         bool    `json:"ok"`
	TTFBMs     float64 `json:"ttfb_ms"`
	TotalMs    float64 `json:"total_ms"`
	MaxGapMs   float64 `json:"max_gap_ms"`
	Err        string  `json:"err"`
}

type group struct {
	n, ok                    int
	inTok, outTok            int64
	first, last              int64
	ttfbOver, totalOver, gap []float64
}

func (g *group) add(r result) {
	if g.n == 0 || r.T < g.first {
		g.first = r.T
	}
	g.last = max(g.last, r.T)
	g.n++
	if !r.OK {
		return
	}
	g.ok++
	g.inTok += int64(r.CtxTokens)
	g.outTok += int64(r.OutTokens)
	planFirst := r.DurPlanMs
	if r.Stream {
		planFirst = r.TTFTPlanMs
		g.gap = append(g.gap, r.MaxGapMs)
	}
	g.ttfbOver = append(g.ttfbOver, r.TTFBMs-planFirst)
	g.totalOver = append(g.totalOver, r.TotalMs-r.DurPlanMs)
}

func pct(xs []float64, p float64) string {
	if len(xs) == 0 {
		return "-"
	}
	sort.Float64s(xs)
	i := int(math.Round(p / 100 * float64(len(xs)-1)))
	return fmt.Sprintf("%.1f", xs[i])
}

func bucket(ctx int) string {
	switch {
	case ctx < 50_000:
		return "a: <50k"
	case ctx < 200_000:
		return "b: 50k-200k"
	case ctx < 500_000:
		return "c: 200k-500k"
	default:
		return "d: 500k-1M"
	}
}

func main() {
	includeWarmup := flag.Bool("warmup", false, "include warmup requests")
	flag.Parse()
	byClass := map[string]*group{}
	byCtx := map[string]*group{}
	byStage := map[int]*group{}
	errs := map[string]int{}
	total := 0
	for _, path := range flag.Args() {
		f, err := os.Open(path)
		if err != nil {
			log.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			var r result
			if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
				log.Fatalf("%s: %v", path, err)
			}
			if r.Warmup && !*includeWarmup {
				continue
			}
			total++
			mode := "non-stream"
			if r.Stream {
				mode = "stream"
			}
			get(byClass, fmt.Sprintf("%s | %s | %s", r.Class, r.Endpoint, mode)).add(r)
			get(byCtx, fmt.Sprintf("%s | %s", bucket(r.CtxTokens), mode)).add(r)
			if byStage[r.Stage] == nil {
				byStage[r.Stage] = &group{}
			}
			byStage[r.Stage].add(r)
			if !r.OK {
				e := r.Err
				if i := strings.IndexAny(e, "{\n"); i > 0 {
					e = e[:i]
				}
				errs[fmt.Sprintf("%d %s", r.Status, e)]++
			}
		}
		f.Close()
	}
	fmt.Printf("%d requests (warmup excluded: %v)\n\n", total, !*includeWarmup)
	fmt.Println("Overheads are milliseconds added on top of the mock's planned time. First byte = first SSE event for streams, first body byte otherwise.")

	fmt.Println("\nTokens/s counts successful requests by start time: input = context sent, output = tokens the mock was told to return.")
	fmt.Println("\n### By stage\n\n| stage | requests | ok % | arrivals/s | input tok/s | output tok/s | first-byte overhead p50 / p99 | total overhead p50 / p99 |\n|---|---|---|---|---|---|---|---|")
	stages := make([]int, 0, len(byStage))
	for s := range byStage {
		stages = append(stages, s)
	}
	sort.Ints(stages)
	for _, s := range stages {
		g := byStage[s]
		rate, inRate, outRate := 0.0, 0.0, 0.0
		if span := float64(g.last-g.first) / 1000; span > 0 {
			rate = float64(g.n) / span
			inRate, outRate = float64(g.inTok)/span, float64(g.outTok)/span
		}
		fmt.Printf("| %d | %d | %.2f | %.1f | %.0f | %.0f | %s / %s | %s / %s |\n", s, g.n, 100*float64(g.ok)/float64(g.n), rate, inRate, outRate,
			pct(g.ttfbOver, 50), pct(g.ttfbOver, 99), pct(g.totalOver, 50), pct(g.totalOver, 99))
	}
	table("By class", "class | endpoint | mode", byClass)
	table("By context size", "context | mode", byCtx)

	if len(errs) > 0 {
		fmt.Println("\n### Errors\n\n| count | status and error |\n|---|---|")
		type kv struct {
			k string
			v int
		}
		var list []kv
		for k, v := range errs {
			list = append(list, kv{k, v})
		}
		sort.Slice(list, func(i, j int) bool { return list[i].v > list[j].v })
		for i, e := range list {
			if i == 15 {
				break
			}
			fmt.Printf("| %d | %s |\n", e.v, strings.ReplaceAll(e.k, "|", "/"))
		}
	}
}

func get(m map[string]*group, k string) *group {
	if m[k] == nil {
		m[k] = &group{}
	}
	return m[k]
}

func table(title, head string, m map[string]*group) {
	cols := strings.Count(head, "|") + 1
	fmt.Printf("\n### %s\n\n| %s | requests | ok %% | first-byte overhead p50 / p90 / p99 | total overhead p50 / p99 | max stream gap p99 |\n|%s---|---|---|---|---|\n",
		title, head, strings.Repeat("---|", cols))
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		g := m[k]
		fmt.Printf("| %s | %d | %.2f | %s / %s / %s | %s / %s | %s |\n", k, g.n, 100*float64(g.ok)/float64(g.n),
			pct(g.ttfbOver, 50), pct(g.ttfbOver, 90), pct(g.ttfbOver, 99), pct(g.totalOver, 50), pct(g.totalOver, 99), pct(g.gap, 99))
	}
}
