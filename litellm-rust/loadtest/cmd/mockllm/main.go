// mockllm is a fake LLM upstream for load tests. It speaks OpenAI chat completions, Anthropic
// messages, Bedrock's Anthropic invoke API and Bedrock Converse (which Bifrost calls instead of
// invoke), streaming and non-streaming, and takes its timing from a directive
// the load generator embeds in the prompt, so the proxy in between forwards it untouched:
//
//	[[mock ttft_ms=800 dur_ms=300000 out=2000 in=10000 status=200]]
//
// ttft_ms is the delay before the first byte, dur_ms the total response time, out the
// number of output tokens, in the input tokens to report (a quarter of the body's bytes when
// absent, which varies with how each proxy serializes the request), status an HTTP status to
// fail with. Without a directive the
// mock answers at once.
//
// Paths may carry a deployment prefix, /d/<id>/..., which is how one mock process stands in
// for many deployments. Hits are counted per deployment at GET /stats, and -faults makes
// chosen deployments fail, e.g. -faults "chat-0-3=429:1.0,msg-1-0=500:0.2". -count-headers
// counts the requests carrying each named header, by value, so a test can check what a proxy
// forwarded, added or stripped.
//
// Bifrost's Bedrock provider takes a host rather than a base URL and always speaks HTTPS, so
// -tls-addr also serves everything over TLS with a throwaway self-signed certificate, and a
// Bedrock path of the form /model/d/<id>/<model>/... (the deployment carried in the inference
// profile ARN) counts as deployment <id>.
package main

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"math/big"
	mrand "math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type fault struct {
	status int
	rate   float64
}

type stats struct {
	mu       sync.Mutex
	hits     map[string]int64
	faults   map[string]int64
	inflight atomic.Int64
	bytesIn  atomic.Int64
	// headers counts requests by header name, then value, for the -count-headers names.
	headers map[string]map[string]int64
}

func (s *stats) hit(dep string, faulted bool) {
	s.mu.Lock()
	s.hits[dep]++
	if faulted {
		s.faults[dep]++
	}
	s.mu.Unlock()
}

type directive struct {
	ttft   time.Duration
	dur    time.Duration
	out    int
	in     int
	status int
}

var directiveMarker = []byte("[[mock ")

// responseSeq numbers responses; responseRun keeps ids unique across mock restarts.
var (
	responseSeq atomic.Uint64
	responseRun = mrand.Uint32()
)

// findDirective looks in the tail first, where the load generator puts it, and only scans the
// whole body when a proxy has reordered fields.
func findDirective(body []byte) directive {
	d := directive{out: 32, status: 200}
	tail := body
	if len(tail) > 64<<10 {
		tail = tail[len(tail)-64<<10:]
	}
	i := bytes.LastIndex(tail, directiveMarker)
	if i < 0 {
		tail = body
		i = bytes.LastIndex(tail, directiveMarker)
		if i < 0 {
			return d
		}
	}
	rest := tail[i+len(directiveMarker):]
	end := bytes.Index(rest, []byte("]]"))
	if end < 0 {
		return d
	}
	for _, field := range strings.Fields(string(rest[:end])) {
		key, value, ok := strings.Cut(field, "=")
		if !ok {
			continue
		}
		n, err := strconv.Atoi(value)
		if err != nil {
			continue
		}
		switch key {
		case "ttft_ms":
			d.ttft = time.Duration(n) * time.Millisecond
		case "dur_ms":
			d.dur = time.Duration(n) * time.Millisecond
		case "out":
			d.out = max(n, 1)
		case "in":
			d.in = max(n, 0)
		case "status":
			d.status = n
		}
	}
	if d.dur < d.ttft {
		d.dur = d.ttft
	}
	return d
}

func deploymentOf(path string) (string, string) {
	if rest, ok := strings.CutPrefix(path, "/d/"); ok {
		id, tail, _ := strings.Cut(rest, "/")
		return id, "/" + tail
	}
	if rest, ok := strings.CutPrefix(path, "/model/d/"); ok {
		id, tail, _ := strings.Cut(rest, "/")
		return id, "/model/" + tail
	}
	return "default", path
}

// sleep waits for d or until the client goes away; it reports whether the wait completed.
func sleep(r *http.Request, d time.Duration) bool {
	if d <= 0 {
		return true
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-r.Context().Done():
		return false
	}
}

const word = "lorem "

func main() {
	addr := flag.String("addr", ":9000", "listen address")
	faultSpec := flag.String("faults", "", "deployment=status:rate,... failures to inject")
	tokensPerChunk := flag.Int("tokens-per-chunk", 4, "output tokens per streamed chunk")
	countHeaders := flag.String("count-headers", "", "comma-separated request headers to count by value at /stats")
	tlsAddr := flag.String("tls-addr", "", "also serve over TLS here, with a self-signed certificate (for Bifrost's Bedrock provider)")
	flag.Parse()

	faults := map[string]fault{}
	for _, item := range strings.Split(*faultSpec, ",") {
		dep, spec, ok := strings.Cut(strings.TrimSpace(item), "=")
		if !ok {
			continue
		}
		code, rate, _ := strings.Cut(spec, ":")
		status, _ := strconv.Atoi(code)
		r, err := strconv.ParseFloat(rate, 64)
		if err != nil {
			r = 1
		}
		faults[dep] = fault{status: status, rate: r}
	}

	st := &stats{hits: map[string]int64{}, faults: map[string]int64{}, headers: map[string]map[string]int64{}}
	var counted []string
	for _, name := range strings.Split(*countHeaders, ",") {
		if name = strings.ToLower(strings.TrimSpace(name)); name != "" {
			counted = append(counted, name)
			st.headers[name] = map[string]int64{}
		}
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /stats", func(w http.ResponseWriter, r *http.Request) {
		st.mu.Lock()
		defer st.mu.Unlock()
		w.Header().Set("content-type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"hits": st.hits, "faults": st.faults, "headers": st.headers,
			"inflight": st.inflight.Load(), "bytes_in": st.bytesIn.Load(),
		})
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok") })
	mux.HandleFunc("POST /", func(w http.ResponseWriter, r *http.Request) {
		st.inflight.Add(1)
		defer st.inflight.Add(-1)
		dep, path := deploymentOf(r.URL.Path)

		buf := bytes.NewBuffer(make([]byte, 0, max(r.ContentLength, 4096)))
		n, err := io.Copy(buf, r.Body)
		st.bytesIn.Add(n)
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		body := buf.Bytes()
		d := findDirective(body)
		f, faulty := faults[dep]
		injected := faulty && mrand.Float64() < f.rate
		st.hit(dep, injected)
		if len(counted) > 0 {
			st.mu.Lock()
			for _, name := range counted {
				for _, value := range r.Header.Values(name) {
					st.headers[name][value]++
				}
			}
			st.mu.Unlock()
		}

		converseStream := strings.HasSuffix(path, "/converse-stream")
		converse := converseStream || strings.HasSuffix(path, "/converse")
		bedrockStream := converseStream || strings.HasSuffix(path, "/invoke-with-response-stream")
		bedrock := bedrockStream || converse || strings.HasSuffix(path, "/invoke")
		anthropic := bedrock || strings.HasSuffix(path, "/messages")
		if !anthropic && !strings.HasSuffix(path, "/chat/completions") {
			http.Error(w, "unknown path "+path, http.StatusNotFound)
			return
		}
		if injected {
			d.status = f.status
		}
		if d.status != 200 {
			if !sleep(r, d.ttft) {
				return
			}
			writeError(w, anthropic, d.status)
			return
		}
		stream := bedrockStream || bytes.Contains(body, []byte(`"stream":true`)) || bytes.Contains(body, []byte(`"stream": true`))
		usage := bytes.Contains(body, []byte(`"include_usage":true`)) || bytes.Contains(body, []byte(`"include_usage": true`))
		inputTokens := len(body) / 4
		// Unique per response, as providers' are: the Python proxy keys spend logs on it.
		seq := responseSeq.Add(1)
		msgID, chatID := fmt.Sprintf("msg_mock%08x%d", responseRun, seq), fmt.Sprintf("chatcmpl-mock%08x%d", responseRun, seq)
		if d.in > 0 {
			inputTokens = d.in
		}
		// Echo the model like a provider does: Bifrost prices a response by the model it names.
		model := requestModel(body)

		if !stream {
			if !sleep(r, d.dur) {
				return
			}
			w.Header().Set("content-type", "application/json")
			text := strings.Repeat(word, d.out)
			if converse {
				json.NewEncoder(w).Encode(map[string]any{
					"output":     map[string]any{"message": map[string]any{"role": "assistant", "content": []any{map[string]any{"text": text}}}},
					"stopReason": "end_turn",
					"usage":      map[string]any{"inputTokens": inputTokens, "outputTokens": d.out, "totalTokens": inputTokens + d.out},
					"metrics":    map[string]any{"latencyMs": d.dur.Milliseconds()},
				})
			} else if anthropic {
				json.NewEncoder(w).Encode(map[string]any{
					"id": msgID, "type": "message", "role": "assistant", "model": model,
					"content":     []any{map[string]any{"type": "text", "text": text}},
					"stop_reason": "end_turn", "stop_sequence": nil,
					"usage": map[string]any{"input_tokens": inputTokens, "output_tokens": d.out},
				})
			} else {
				json.NewEncoder(w).Encode(map[string]any{
					"id": chatID, "object": "chat.completion", "created": time.Now().Unix(), "model": model,
					"choices": []any{map[string]any{"index": 0, "finish_reason": "stop",
						"message": map[string]any{"role": "assistant", "content": text}}},
					"usage": map[string]any{"prompt_tokens": inputTokens, "completion_tokens": d.out, "total_tokens": inputTokens + d.out},
				})
			}
			return
		}

		if !sleep(r, d.ttft) {
			return
		}
		if bedrock {
			w.Header().Set("content-type", "application/vnd.amazon.eventstream")
		} else {
			w.Header().Set("content-type", "text/event-stream")
		}
		w.Header().Set("cache-control", "no-cache")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		chunks := max((d.out+*tokensPerChunk-1) / *tokensPerChunk, 1)
		interval := (d.dur - d.ttft) / time.Duration(chunks)
		piece := strings.Repeat(word, *tokensPerChunk)
		start := time.Now()

		var event func(name string, data any)
		if converse {
			// Converse events are typed by the frame's :event-type, with the JSON as the payload.
			event = func(name string, data any) {
				b, _ := json.Marshal(data)
				w.Write(eventStreamFrame(name, b))
				flusher.Flush()
			}
		} else if bedrock {
			event = func(_ string, data any) {
				b, _ := json.Marshal(data)
				payload, _ := json.Marshal(map[string]string{"bytes": base64.StdEncoding.EncodeToString(b)})
				w.Write(eventStreamFrame("chunk", payload))
				flusher.Flush()
			}
		} else if anthropic {
			event = func(name string, data any) {
				b, _ := json.Marshal(data)
				fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, b)
				flusher.Flush()
			}
		} else {
			event = func(_ string, data any) {
				b, _ := json.Marshal(data)
				fmt.Fprintf(w, "data: %s\n\n", b)
				flusher.Flush()
			}
		}
		if converse {
			event("messageStart", map[string]any{"role": "assistant"})
		} else if anthropic {
			event("message_start", map[string]any{"type": "message_start", "message": map[string]any{
				"id": msgID, "type": "message", "role": "assistant", "model": model, "content": []any{},
				"stop_reason": nil, "stop_sequence": nil, "usage": map[string]any{"input_tokens": inputTokens, "output_tokens": 1}}})
			event("content_block_start", map[string]any{"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "text", "text": ""}})
		}
		created := time.Now().Unix()
		openaiChunk := func(delta map[string]any, finish any) map[string]any {
			return map[string]any{"id": chatID, "object": "chat.completion.chunk", "created": created, "model": model,
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
		}
		for i := range chunks {
			// Pace against the start time so slow writes do not stretch the stream.
			if !sleep(r, time.Until(start.Add(interval*time.Duration(i)))) {
				return
			}
			if converse {
				event("contentBlockDelta", map[string]any{"contentBlockIndex": 0, "delta": map[string]any{"text": piece}})
			} else if anthropic {
				event("content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
					"delta": map[string]any{"type": "text_delta", "text": piece}})
			} else {
				event("", openaiChunk(map[string]any{"content": piece}, nil))
			}
		}
		if !sleep(r, time.Until(start.Add(d.dur-d.ttft))) {
			return
		}
		if converse {
			event("contentBlockStop", map[string]any{"contentBlockIndex": 0})
			event("messageStop", map[string]any{"stopReason": "end_turn"})
			event("metadata", map[string]any{
				"usage":   map[string]any{"inputTokens": inputTokens, "outputTokens": d.out, "totalTokens": inputTokens + d.out},
				"metrics": map[string]any{"latencyMs": d.dur.Milliseconds()}})
			return
		}
		if anthropic {
			event("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
			event("message_delta", map[string]any{"type": "message_delta",
				"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
				"usage": map[string]any{"output_tokens": d.out}})
			event("message_stop", map[string]any{"type": "message_stop"})
			return
		}
		event("", openaiChunk(map[string]any{}, "stop"))
		if usage {
			event("", map[string]any{"id": chatID, "object": "chat.completion.chunk", "created": created, "model": model,
				"choices": []any{}, "usage": map[string]any{"prompt_tokens": inputTokens, "completion_tokens": d.out, "total_tokens": inputTokens + d.out}})
		}
		io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	})

	server := &http.Server{Addr: *addr, Handler: mux, IdleTimeout: 15 * time.Minute, ReadHeaderTimeout: 30 * time.Second}
	if *tlsAddr != "" {
		tlsServer := &http.Server{Addr: *tlsAddr, Handler: mux, IdleTimeout: 15 * time.Minute, ReadHeaderTimeout: 30 * time.Second,
			TLSConfig: &tls.Config{Certificates: []tls.Certificate{selfSigned()}}}
		go func() { log.Fatal(tlsServer.ListenAndServeTLS("", "")) }()
		log.Printf("mockllm listening on %s (TLS)", *tlsAddr)
	}
	log.Printf("mockllm listening on %s (faults: %v)", *addr, faults)
	log.Fatal(server.ListenAndServe())
}

func writeError(w http.ResponseWriter, anthropic bool, status int) {
	w.Header().Set("content-type", "application/json")
	if status == http.StatusTooManyRequests {
		w.Header().Set("retry-after", "1")
	}
	w.WriteHeader(status)
	if anthropic {
		kind := "api_error"
		if status == http.StatusTooManyRequests {
			kind = "rate_limit_error"
		}
		json.NewEncoder(w).Encode(map[string]any{"type": "error", "error": map[string]any{"type": kind, "message": "mock failure"}})
		return
	}
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": "mock failure", "type": "server_error", "code": status}})
}

// eventStreamFrame encodes one AWS event-stream message, the framing of Bedrock's
// invoke-with-response-stream and converse-stream: prelude (lengths + CRC), headers, payload,
// message CRC.
func eventStreamFrame(eventType string, payload []byte) []byte {
	var headers bytes.Buffer
	for _, h := range [][2]string{{":event-type", eventType}, {":content-type", "application/json"}, {":message-type", "event"}} {
		headers.WriteByte(byte(len(h[0])))
		headers.WriteString(h[0])
		headers.WriteByte(7)
		binary.Write(&headers, binary.BigEndian, uint16(len(h[1])))
		headers.WriteString(h[1])
	}
	total := 12 + headers.Len() + len(payload) + 4
	frame := make([]byte, 0, total)
	frame = binary.BigEndian.AppendUint32(frame, uint32(total))
	frame = binary.BigEndian.AppendUint32(frame, uint32(headers.Len()))
	frame = binary.BigEndian.AppendUint32(frame, crc32.ChecksumIEEE(frame))
	frame = append(frame, headers.Bytes()...)
	frame = append(frame, payload...)
	return binary.BigEndian.AppendUint32(frame, crc32.ChecksumIEEE(frame))
}

// selfSigned makes a throwaway certificate; clients of -tls-addr skip verification.
func selfSigned() tls.Certificate {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "mockllm"},
		DNSNames: []string{"*"}, NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(365 * 24 * time.Hour)}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		log.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
}

// requestModel is the request body's top-level "model" string, or "mock" when it has none
// (Bedrock carries the model in the path). A cheap scan: model is a short field near the start.
func requestModel(body []byte) string {
	i := bytes.Index(body, []byte(`"model"`))
	if i < 0 {
		return "mock"
	}
	rest, ok := bytes.CutPrefix(bytes.TrimLeft(body[i+len(`"model"`):], " \t\r\n"), []byte(":"))
	if !ok {
		return "mock"
	}
	if end := bytes.IndexAny(rest, ",}"); end > 0 {
		var model string
		if json.Unmarshal(bytes.TrimSpace(rest[:end]), &model) == nil && model != "" {
			return model
		}
	}
	return "mock"
}
