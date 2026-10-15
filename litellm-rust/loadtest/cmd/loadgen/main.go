// loadgen is an open-loop load generator for the LiteLLM gateway. Requests arrive as a Poisson
// process (so a slow proxy cannot slow the arrivals down, unlike a closed loop), each drawn from a
// weighted class with its own endpoint, stream share, context size and simulated model time. The
// simulated timing travels to mockllm as a directive in the prompt, so every result line carries
// both what the mock was told to take and what the client saw; the difference is proxy overhead.
//
// Run several copies on separate machines with -shard i -shards n; each sends rate/n.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type result struct {
	T          int64   `json:"t"`
	Shard      int     `json:"shard"`
	Stage      int     `json:"stage"`
	Warmup     bool    `json:"warmup,omitempty"`
	Class      string  `json:"class"`
	Endpoint   string  `json:"ep"`
	Model      string  `json:"model"`
	KeyIdx     int     `json:"key"`
	Stream     bool    `json:"stream"`
	NewConn    bool    `json:"new_conn,omitempty"`
	CtxTokens  int     `json:"ctx"`
	BodyBytes  int     `json:"bytes"`
	TTFTPlanMs int64   `json:"ttft_plan_ms"`
	DurPlanMs  int64   `json:"dur_plan_ms"`
	Status     int     `json:"status"`
	OK         bool    `json:"ok"`
	TTFBMs     float64 `json:"ttfb_ms"`
	TotalMs    float64 `json:"total_ms"`
	MaxGapMs   float64 `json:"max_gap_ms,omitempty"`
	Chunks     int     `json:"chunks,omitempty"`
	Err        string  `json:"err,omitempty"`
}

type counters struct {
	sent, ok, failed, shed, inflight atomic.Int64
}

func main() {
	workloadPath := flag.String("workload", "", "workload JSON file")
	target := flag.String("target", "http://127.0.0.1:4000", "gateway base URL")
	out := flag.String("out", "results.jsonl", "per-request results (JSONL)")
	shard := flag.Int("shard", 0, "this generator's index")
	shards := flag.Int("shards", 1, "number of generators sharing the workload")
	key := flag.String("key", "", "use this one key for every request instead of the keys file")
	messagesAuth := flag.String("messages-auth", "bearer", "how /v1/messages sends the key: bearer or x-api-key")
	seed := flag.Uint64("seed", 1, "random seed (combined with the shard index)")
	flag.Parse()

	w, err := loadWorkload(*workloadPath)
	if err != nil {
		log.Fatal(err)
	}
	keys := []string{*key}
	if *key == "" {
		keys, err = readLines(w.Keys.File)
		if err != nil {
			log.Fatalf("keys: %v (or pass -key)", err)
		}
	}
	pool := newTextPool(8<<20, *seed)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	f, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	results := make(chan result, 4096)
	var writerDone sync.WaitGroup
	writerDone.Add(1)
	go func() {
		defer writerDone.Done()
		bw := bufio.NewWriterSize(f, 1<<20)
		enc := json.NewEncoder(bw)
		tick := time.NewTicker(time.Second)
		defer tick.Stop()
		for {
			select {
			case r, ok := <-results:
				if !ok {
					bw.Flush()
					f.Close()
					return
				}
				enc.Encode(r)
			case <-tick.C:
				bw.Flush()
			}
		}
	}()

	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	pooled := &http.Transport{
		DialContext: dialer.DialContext, MaxIdleConns: 0, MaxIdleConnsPerHost: 4096,
		IdleConnTimeout: 90 * time.Second, DisableCompression: true,
		WriteBufferSize: 256 << 10, ReadBufferSize: 64 << 10,
	}
	fresh := &http.Transport{
		DialContext: dialer.DialContext, DisableKeepAlives: true, DisableCompression: true,
		WriteBufferSize: 256 << 10, ReadBufferSize: 64 << 10,
	}

	rng := rand.New(rand.NewPCG(*seed, uint64(*shard)))
	var zipf *rand.Zipf
	if w.Keys.ZipfS > 1 && len(keys) > 1 {
		zipf = rand.NewZipf(rng, w.Keys.ZipfS, 1, uint64(len(keys)-1))
	}
	classWeights := make([]float64, len(w.Classes))
	for i, c := range w.Classes {
		classWeights[i] = c.Weight
	}

	var c counters
	var wg sync.WaitGroup
	progress := time.NewTicker(10 * time.Second)
	defer progress.Stop()
	go func() {
		for range progress.C {
			log.Printf("sent=%d ok=%d failed=%d shed=%d inflight=%d",
				c.sent.Load(), c.ok.Load(), c.failed.Load(), c.shed.Load(), c.inflight.Load())
		}
	}()

	// Requests get their own context so the final drain can outlive the schedule.
	reqCtx, cancelRequests := context.WithCancel(context.Background())
	defer cancelRequests()

	start := time.Now()
	next := start
	stage, stageEnd := 0, start.Add(seconds(w.Stages[0].DurationS+w.WarmupS))
	warmEnd := start.Add(seconds(w.WarmupS))
	id := 0
schedule:
	for {
		for !next.Before(stageEnd) {
			stage++
			if stage == len(w.Stages) {
				break schedule
			}
			stageEnd = stageEnd.Add(seconds(w.Stages[stage].DurationS))
		}
		rate := w.Stages[stage].RPS / float64(*shards)
		if rate <= 0 {
			next = stageEnd
			continue
		}
		next = next.Add(seconds(rng.ExpFloat64() / rate))
		select {
		case <-ctx.Done():
			break schedule
		case <-time.After(time.Until(next)):
		}
		if c.inflight.Load() >= int64(w.MaxInFlight) {
			c.shed.Add(1)
			continue
		}
		cl := w.Classes[weighted(rng, classWeights, len(w.Classes))]
		models := w.Models[cl.Endpoint]
		keyIdx := 0
		if zipf != nil {
			keyIdx = int(zipf.Uint64())
		} else if len(keys) > 1 {
			keyIdx = rng.IntN(len(keys))
		}
		durS := cl.DurationS.Sample(rng)
		ttftMs := int64(cl.TTFTMs.Sample(rng))
		durMs := max(int64(durS*1000), ttftMs)
		tps := cl.OutputTokensPerS
		if tps == 0 {
			tps = 50
		}
		res := result{
			T: time.Now().UnixMilli(), Shard: *shard, Stage: stage, Warmup: time.Now().Before(warmEnd),
			Class: cl.Name, Endpoint: cl.Endpoint,
			Model:     fmt.Sprintf("%s%d", models.Prefix, rng.IntN(models.Count)),
			KeyIdx:    keyIdx,
			Stream:    rng.Float64() < cl.StreamFraction,
			NewConn:   rng.Float64() < w.NewConnFraction,
			CtxTokens: int(cl.ContextTokens.Sample(rng)), TTFTPlanMs: ttftMs, DurPlanMs: durMs,
		}
		outTokens := max(int(float64(durMs-ttftMs)/1000*tps), 1)
		id++
		body := buildBody(pool, rng, res, outTokens, fmt.Sprintf("%d-%d", *shard, id))
		res.BodyBytes = len(body)
		transport := pooled
		if res.NewConn {
			transport = fresh
		}
		c.sent.Add(1)
		c.inflight.Add(1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer c.inflight.Add(-1)
			do(reqCtx, transport, *target, keys[keyIdx], *messagesAuth, body, &res)
			if res.OK {
				c.ok.Add(1)
			} else {
				c.failed.Add(1)
			}
			results <- res
		}()
	}

	drain := seconds(w.longest() + 120)
	log.Printf("schedule finished after %s; draining %d in-flight requests (up to %s)", time.Since(start).Round(time.Second), c.inflight.Load(), drain)
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(drain):
		log.Printf("drain timed out; cancelling %d requests", c.inflight.Load())
		cancelRequests()
		<-done
	case <-ctx.Done():
		cancelRequests()
		<-done
	}
	close(results)
	writerDone.Wait()
	log.Printf("done: sent=%d ok=%d failed=%d shed=%d", c.sent.Load(), c.ok.Load(), c.failed.Load(), c.shed.Load())
}

func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

func do(ctx context.Context, transport *http.Transport, target, key, messagesAuth string, body []byte, res *result) {
	limit := time.Duration(res.DurPlanMs)*time.Millisecond + 3*time.Minute
	ctx, cancel := context.WithTimeout(ctx, limit)
	defer cancel()

	path := "/v1/chat/completions"
	if res.Endpoint == "messages" {
		path = "/v1/messages"
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, target+path, bytes.NewReader(body))
	req.Header.Set("content-type", "application/json")
	if res.Endpoint == "messages" {
		req.Header.Set("anthropic-version", "2023-06-01")
		if messagesAuth == "x-api-key" {
			req.Header.Set("x-api-key", key)
		} else {
			req.Header.Set("authorization", "Bearer "+key)
		}
	} else {
		req.Header.Set("authorization", "Bearer "+key)
	}

	t0 := time.Now()
	resp, err := transport.RoundTrip(req)
	if err != nil {
		res.Err = trim(err.Error())
		res.TotalMs = ms(time.Since(t0))
		return
	}
	defer resp.Body.Close()
	res.Status = resp.StatusCode

	if !res.Stream || resp.StatusCode != http.StatusOK {
		first := make([]byte, 1)
		n, _ := io.ReadFull(resp.Body, first)
		res.TTFBMs = ms(time.Since(t0))
		rest, err := io.ReadAll(resp.Body)
		res.TotalMs = ms(time.Since(t0))
		full := append(first[:n], rest...)
		marker := []byte(`"choices"`)
		if res.Endpoint == "messages" {
			marker = []byte(`"content"`)
		}
		res.OK = err == nil && resp.StatusCode == http.StatusOK && bytes.Contains(full, marker)
		if !res.OK {
			res.Err = trim(fmt.Sprintf("status %d: %s", resp.StatusCode, full))
			if err != nil {
				res.Err = trim(err.Error())
			}
		}
		return
	}

	reader := bufio.NewReaderSize(resp.Body, 64<<10)
	var last time.Time
	var gap time.Duration
	done := false
	for {
		line, err := reader.ReadSlice('\n')
		if bytes.HasPrefix(line, []byte("data:")) {
			now := time.Now()
			if res.Chunks == 0 {
				res.TTFBMs = ms(now.Sub(t0))
			} else {
				gap = max(gap, now.Sub(last))
			}
			last = now
			res.Chunks++
			if bytes.Contains(line, []byte("[DONE]")) || bytes.Contains(line, []byte(`"message_stop"`)) {
				done = true
			}
		}
		if err == bufio.ErrBufferFull {
			continue
		}
		if err != nil {
			if err != io.EOF {
				res.Err = trim(err.Error())
			}
			break
		}
	}
	res.TotalMs = ms(time.Since(t0))
	res.MaxGapMs = ms(gap)
	res.OK = done && res.Err == ""
	if !done && res.Err == "" {
		res.Err = "stream ended without a final event"
	}
}

func ms(d time.Duration) float64 { return math.Round(float64(d.Microseconds())/10) / 100 }

func trim(s string) string {
	if len(s) > 300 {
		return s[:300]
	}
	return s
}

func readLines(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, line := range strings.Split(string(b), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			out = append(out, line)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s is empty", path)
	}
	return out, nil
}
