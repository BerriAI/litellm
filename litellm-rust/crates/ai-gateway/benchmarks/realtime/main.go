// WebSocket Realtime API benchmark — measures dial, session establishment, and first audio delta latency.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"os"
	"sort"
	"sync"
	"time"

	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
)

const defaultModel = "gpt-realtime-2"

// Set in main() from flags so the same binary can target OpenAI directly or the
// local gateway (ws:// when -insecure).
var (
	apiKey string
	wsURL  string
)

type Result struct {
	ConnID       int
	Success      bool
	DialMS       float64
	SessionMS    float64
	FirstAudioMS float64
	TotalMS      float64
	Error        string
}

var verbose bool

func logEvent(connID int, direction string, event map[string]any) {
	if !verbose {
		return
	}
	etype, _ := event["type"].(string)
	if _, hasDelta := event["delta"]; hasDelta {
		if s, ok := event["delta"].(string); ok && len(s) > 40 {
			event["delta"] = s[:40] + "…"
		}
	}
	b, _ := json.Marshal(event)
	fmt.Printf("  [conn %d] %s %s: %s\n", connID, direction, etype, b)
}

func runConnection(connID int, model string, timeout time.Duration) Result {
	overallStart := time.Now()

	wsEndpoint := wsURL + "?model=" + url.QueryEscape(model)

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	dialStart := time.Now()
	conn, _, err := websocket.Dial(ctx, wsEndpoint, &websocket.DialOptions{
		HTTPHeader: http.Header{
			"Authorization": []string{"Bearer " + apiKey},
		},
	})
	if err != nil {
		return Result{ConnID: connID, Success: false, Error: fmt.Sprintf("[conn %d] dial: %v", connID, err)}
	}
	defer conn.CloseNow()
	dialMS := ms(time.Since(dialStart))

	// Wait for session.created
	sessionStart := time.Now()
	sessionMS, err := waitForSession(ctx, conn, connID)
	if err != nil {
		return Result{ConnID: connID, Success: false, Error: fmt.Sprintf("[conn %d] session: %v", connID, err)}
	}
	_ = sessionStart

	// Send conversation item
	itemMsg := map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type": "message",
			"role": "user",
			"content": []map[string]any{
				{"type": "input_text", "text": "Say hi."},
			},
		},
	}
	logEvent(connID, ">>", itemMsg)
	if err := wsjson.Write(ctx, conn, itemMsg); err != nil {
		return Result{ConnID: connID, Success: false, Error: fmt.Sprintf("[conn %d] send item: %v", connID, err)}
	}

	responseMsg := map[string]any{"type": "response.create"}
	logEvent(connID, ">>", responseMsg)
	if err := wsjson.Write(ctx, conn, responseMsg); err != nil {
		return Result{ConnID: connID, Success: false, Error: fmt.Sprintf("[conn %d] send response.create: %v", connID, err)}
	}

	firstAudioMS, err := waitForFirstAudio(ctx, conn, connID)
	if err != nil {
		return Result{ConnID: connID, Success: false, Error: fmt.Sprintf("[conn %d] audio: %v", connID, err)}
	}

	conn.Close(websocket.StatusNormalClosure, "")
	totalMS := ms(time.Since(overallStart))

	return Result{
		ConnID:       connID,
		Success:      true,
		DialMS:       dialMS,
		SessionMS:    sessionMS,
		FirstAudioMS: firstAudioMS,
		TotalMS:      totalMS,
	}
}

func waitForSession(ctx context.Context, conn *websocket.Conn, connID int) (float64, error) {
	t := time.Now()
	for {
		var event map[string]any
		if err := wsjson.Read(ctx, conn, &event); err != nil {
			return 0, err
		}
		logEvent(connID, "<<", event)
		switch event["type"] {
		case "session.created":
			return ms(time.Since(t)), nil
		case "error":
			return 0, fmt.Errorf("%v", event)
		}
	}
}

func waitForFirstAudio(ctx context.Context, conn *websocket.Conn, connID int) (float64, error) {
	t := time.Now()
	for {
		var event map[string]any
		if err := wsjson.Read(ctx, conn, &event); err != nil {
			return 0, err
		}
		logEvent(connID, "<<", event)
		switch event["type"] {
		case "response.output_audio.delta":
			return ms(time.Since(t)), nil
		case "error":
			return 0, fmt.Errorf("%v", event)
		case "response.done":
			resp, _ := event["response"].(map[string]any)
			output, _ := resp["output"].([]any)
			var types []string
			for _, item := range output {
				m, _ := item.(map[string]any)
				for _, c := range toSlice(m["content"]) {
					cm, _ := c.(map[string]any)
					if t, ok := cm["type"].(string); ok {
						types = append(types, t)
					}
				}
			}
			return 0, fmt.Errorf("response.done with no audio delta — output types: %v", types)
		}
	}
}

func toSlice(v any) []any {
	if s, ok := v.([]any); ok {
		return s
	}
	return nil
}

func ms(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}

func percentile(sorted []float64, p int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(float64(len(sorted))*float64(p)/100.0)) - 1
	if idx < 0 {
		idx = 0
	}
	return sorted[idx]
}

func median(sorted []float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func printSummary(results []Result, model string) {
	sep := "────────────────────────────────────────────────────────────"
	fmt.Printf("\n%s\n  SUMMARY\n%s\n", sep, sep)
	fmt.Printf("  Target : %s\n", wsURL)
	fmt.Printf("  Model  : %s\n", model)
	fmt.Println(sep)

	var successes []Result
	for _, r := range results {
		if r.Success {
			successes = append(successes, r)
		}
	}
	fmt.Printf("  SUCCESS          %d/%d\n", len(successes), len(results))

	if len(successes) == 0 {
		fmt.Println("\n  All connections failed.")
		for _, r := range results {
			fmt.Printf("    %s\n", r.Error)
		}
		return
	}

	totals := make([]float64, len(successes))
	dials := make([]float64, len(successes))
	audios := make([]float64, len(successes))
	for i, r := range successes {
		totals[i] = r.TotalMS
		dials[i] = r.DialMS
		audios[i] = r.FirstAudioMS
	}
	sort.Float64s(totals)
	sort.Float64s(dials)
	sort.Float64s(audios)

	fmt.Printf("  MEDIAN TOTAL     %.0f ms\n", median(totals))
	fmt.Printf("  AVG TOTAL        %.0f ms\n", mean(totals))
	fmt.Printf("  P95 TOTAL        %.0f ms\n", percentile(totals, 95))
	fmt.Printf("  MEDIAN DIAL      %.0f ms\n", median(dials))
	fmt.Printf("  MEDIAN 1ST AUDIO %.0f ms\n", median(audios))
	fmt.Println(sep)

	var failures []Result
	for _, r := range results {
		if !r.Success {
			failures = append(failures, r)
		}
	}
	if len(failures) > 0 {
		fmt.Println("\n  Failures:")
		for _, r := range failures {
			fmt.Printf("    %s\n", r.Error)
		}
	}

	fmt.Printf("\n  Per-connection breakdown:\n")
	fmt.Printf("  %4s  %8s  %10s  %10s  %8s\n", "#", "dial", "session", "1st audio", "total")
	fmt.Printf("  %4s  %8s  %10s  %10s  %8s\n", "────", "────────", "──────────", "──────────", "────────")
	for _, r := range results {
		if r.Success {
			fmt.Printf("  %4d  %6.0fms  %8.0fms  %8.0fms  %6.0fms\n",
				r.ConnID, r.DialMS, r.SessionMS, r.FirstAudioMS, r.TotalMS)
		} else {
			fmt.Printf("  %4d  FAILED — %s\n", r.ConnID, r.Error)
		}
	}
}

func main() {
	n := flag.Int("n", 5, "Number of WebSocket connections to open")
	model := flag.String("m", defaultModel, "Realtime model to use")
	timeoutSec := flag.Float64("t", 30.0, "Per-connection timeout in seconds")
	concurrency := flag.Int("c", 10, "Max simultaneous connections")
	host := flag.String("host", "api.openai.com", "Target host[:port] for /v1/realtime")
	key := flag.String("key", os.Getenv("OPENAI_API_KEY"), "Bearer token (defaults to $OPENAI_API_KEY)")
	insecure := flag.Bool("insecure", false, "Use ws:// instead of wss:// (e.g. local gateway)")
	flag.BoolVar(&verbose, "v", false, "Print every WebSocket event sent and received")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), `Usage: ws_benchmark [options]

Options:
`)
		flag.PrintDefaults()
		fmt.Fprintf(flag.CommandLine.Output(), `
Metrics measured
────────────────
  WS Dial          TCP + TLS + HTTP→WebSocket upgrade handshake
  Session          handshake complete → session.created event received
  1st Audio Delta  response.create sent → first response.output_audio.delta received
  Total            full wall-clock time per connection

Examples
────────
  # Run with defaults (5 connections, concurrency 10)
  ./ws_benchmark

  # 20 connections, max 5 running at once
  ./ws_benchmark -n 20 -c 5

  # 45 connections with verbose event logging
  ./ws_benchmark -n 45 -c 10 -v

  # Override model and timeout
  ./ws_benchmark -n 10 -m gpt-realtime-2 -t 60
`)
	}
	flag.Parse()

	scheme := "wss"
	if *insecure {
		scheme = "ws"
	}
	wsURL = scheme + "://" + *host + "/v1/realtime"
	apiKey = *key

	timeout := time.Duration(*timeoutSec * float64(time.Second))
	fmt.Printf("Running %d connection(s) (concurrency=%d, timeout=%.0fs) …\n", *n, *concurrency, *timeoutSec)

	results := make([]Result, *n)
	sem := make(chan struct{}, *concurrency)
	var wg sync.WaitGroup

	for i := 0; i < *n; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			sem <- struct{}{}
			results[id-1] = runConnection(id, *model, timeout)
			<-sem
		}(i + 1)
	}
	wg.Wait()

	printSummary(results, *model)
}
