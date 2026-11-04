package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
)

// upstream is one request a proxy sent to the mock provider.
type upstream struct {
	Method  string            `json:"method"`
	Path    string            `json:"path"`
	Query   string            `json:"query,omitempty"`
	Headers map[string]string `json:"headers"`
	Body    any               `json:"body,omitempty"`
}

// mock answers as each provider would, the same way for both proxies, and records what it was
// sent. Cases run one at a time, so everything that arrives belongs to the current slot.
type mock struct {
	mu      sync.Mutex
	current []upstream
}

func (m *mock) take() []upstream {
	m.mu.Lock()
	defer m.mu.Unlock()
	got := m.current
	m.current = nil
	return got
}

func (m *mock) record(r *http.Request, body []byte) {
	headers := map[string]string{}
	for name, values := range r.Header {
		headers[strings.ToLower(name)] = strings.Join(values, ", ")
	}
	var parsed any
	if len(body) > 0 && json.Unmarshal(body, &parsed) != nil {
		parsed = string(body)
	}
	m.mu.Lock()
	m.current = append(m.current, upstream{r.Method, r.URL.Path, r.URL.RawQuery, headers, parsed})
	m.mu.Unlock()
}

// The fixed answer parts. Everything the mock returns is deterministic, so any difference in
// what the clients get comes from the proxies.
const (
	answerText     = "Hello from the parity mock."
	reasoningText  = "The user wants a greeting."
	signature      = "sig_parity"
	toolID         = "call_parity_1"
	toolArgs       = `{"city": "Paris"}`
	jsonAnswer     = `{"answer": 42}`
	inputTokens    = 21
	outputTokens   = 9
	cacheRead      = 3
	cacheWrite     = 2
	reasoningCount = 4
)

// pixel is a 1x1 PNG, served at /img.png for remote image cases.
var pixel, _ = base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

var directive = regexp.MustCompile(`\[\[parity reply=(\w+)\]\]`)

// reply is what a case asks the provider to answer with: text, tool, thinking or json.
type reply struct {
	kind string
	tool string // the tool to call: the request's first tool, or the proxy's JSON-mode tool
}

func replyFor(body []byte, firstTool string) reply {
	kind := "text"
	if m := directive.FindSubmatch(body); m != nil {
		kind = string(m[1])
	}
	// Anthropic-style JSON mode turns response_format into a forced tool; answer through it.
	if bytes.Contains(body, []byte(`"json_tool_call"`)) {
		return reply{kind: "tool", tool: "json_tool_call"}
	}
	return reply{kind: kind, tool: firstTool}
}

func (r reply) text() string {
	if r.kind == "json" {
		return jsonAnswer
	}
	return answerText
}

func (m *mock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	path := r.URL.Path
	switch {
	case path == "/img.png":
		w.Header().Set("Content-Type", "image/png")
		w.Write(pixel)
		return
	case path == "/token":
		writeJSON(w, map[string]any{"access_token": "ya29.parity", "expires_in": 3600, "token_type": "Bearer"})
		return
	}
	m.record(r, body)
	var req map[string]any
	json.Unmarshal(body, &req)
	stream, _ := req["stream"].(bool)
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		openAI(w, req, replyFor(body, str(dig(req, "tools", 0, "function", "name"))), stream)
	case strings.HasSuffix(path, "/messages"), strings.HasSuffix(path, ":rawPredict"):
		anthropic(w, replyFor(body, str(dig(req, "tools", 0, "name"))), stream)
	case strings.HasSuffix(path, ":streamRawPredict"):
		anthropic(w, replyFor(body, str(dig(req, "tools", 0, "name"))), true)
	case strings.HasSuffix(path, "/converse"):
		bedrock(w, replyFor(body, str(dig(req, "toolConfig", "tools", 0, "toolSpec", "name"))), false)
	case strings.HasSuffix(path, "/converse-stream"):
		bedrock(w, replyFor(body, str(dig(req, "toolConfig", "tools", 0, "toolSpec", "name"))), true)
	case strings.HasSuffix(path, ":generateContent"):
		gemini(w, replyFor(body, str(dig(req, "tools", 0, "functionDeclarations", 0, "name"))), false)
	case strings.HasSuffix(path, ":streamGenerateContent"):
		gemini(w, replyFor(body, str(dig(req, "tools", 0, "functionDeclarations", 0, "name"))), true)
	default:
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, map[string]any{"error": map[string]any{"message": "parity mock: no route for " + path}})
	}
}

func openAI(w http.ResponseWriter, req map[string]any, rep reply, stream bool) {
	model, _ := req["model"].(string)
	usage := map[string]any{
		"prompt_tokens": inputTokens, "completion_tokens": outputTokens, "total_tokens": inputTokens + outputTokens,
		"prompt_tokens_details": map[string]any{"cached_tokens": cacheRead},
	}
	message := map[string]any{"role": "assistant", "content": rep.text()}
	finish := "stop"
	if rep.kind == "tool" {
		message["content"] = nil
		message["tool_calls"] = []any{map[string]any{"id": toolID, "type": "function",
			"function": map[string]any{"name": rep.tool, "arguments": toolArgs}}}
		finish = "tool_calls"
	}
	if rep.kind == "thinking" {
		message["reasoning_content"] = reasoningText
	}
	if !stream {
		writeJSON(w, map[string]any{"id": "chatcmpl-parity", "object": "chat.completion", "created": 1700000000,
			"model": model, "choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finish}},
			"usage": usage})
		return
	}
	s := newSSE(w, "text/event-stream")
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{"id": "chatcmpl-parity", "object": "chat.completion.chunk", "created": 1700000000,
			"model": model, "choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
	}
	s.data(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
	if rep.kind == "thinking" {
		s.data(chunk(map[string]any{"reasoning_content": reasoningText}, nil))
	}
	if rep.kind == "tool" {
		s.data(chunk(map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": toolID, "type": "function",
			"function": map[string]any{"name": rep.tool, "arguments": ""}}}}, nil))
		for _, part := range halves(toolArgs) {
			s.data(chunk(map[string]any{"tool_calls": []any{map[string]any{"index": 0,
				"function": map[string]any{"arguments": part}}}}, nil))
		}
	} else {
		for _, part := range halves(rep.text()) {
			s.data(chunk(map[string]any{"content": part}, nil))
		}
	}
	s.data(chunk(map[string]any{}, finish))
	if b, _ := dig(req, "stream_options", "include_usage").(bool); b {
		last := chunk(nil, nil)
		last["choices"] = []any{}
		last["usage"] = usage
		s.data(last)
	}
	s.raw("data: [DONE]\n\n")
}

func anthropic(w http.ResponseWriter, rep reply, stream bool) {
	var blocks []map[string]any
	stop := "end_turn"
	if rep.kind == "thinking" {
		blocks = append(blocks, map[string]any{"type": "thinking", "thinking": reasoningText, "signature": signature})
	}
	if rep.kind == "tool" {
		var input any
		json.Unmarshal([]byte(toolArgs), &input)
		if rep.tool == "json_tool_call" {
			json.Unmarshal([]byte(jsonAnswer), &input)
		}
		blocks = append(blocks, map[string]any{"type": "tool_use", "id": toolID, "name": rep.tool, "input": input})
		stop = "tool_use"
	} else {
		blocks = append(blocks, map[string]any{"type": "text", "text": rep.text()})
	}
	usage := map[string]any{"input_tokens": inputTokens, "output_tokens": outputTokens,
		"cache_read_input_tokens": cacheRead, "cache_creation_input_tokens": cacheWrite}
	if !stream {
		content := make([]any, len(blocks))
		for i, b := range blocks {
			content[i] = b
		}
		writeJSON(w, map[string]any{"id": "msg_parity", "type": "message", "role": "assistant",
			"model": "claude-parity", "content": content, "stop_reason": stop, "stop_sequence": nil, "usage": usage})
		return
	}
	s := newSSE(w, "text/event-stream")
	s.event("message_start", map[string]any{"type": "message_start", "message": map[string]any{
		"id": "msg_parity", "type": "message", "role": "assistant", "model": "claude-parity", "content": []any{},
		"stop_reason": nil, "stop_sequence": nil, "usage": map[string]any{"input_tokens": inputTokens, "output_tokens": 1,
			"cache_read_input_tokens": cacheRead, "cache_creation_input_tokens": cacheWrite}}})
	for i, b := range blocks {
		start := map[string]any{}
		for k, v := range b {
			start[k] = v
		}
		var deltas []map[string]any
		switch b["type"] {
		case "thinking":
			start["thinking"], start["signature"] = "", ""
			for _, part := range halves(reasoningText) {
				deltas = append(deltas, map[string]any{"type": "thinking_delta", "thinking": part})
			}
			deltas = append(deltas, map[string]any{"type": "signature_delta", "signature": signature})
		case "tool_use":
			start["input"] = map[string]any{}
			input, _ := json.Marshal(b["input"])
			for _, part := range halves(string(input)) {
				deltas = append(deltas, map[string]any{"type": "input_json_delta", "partial_json": part})
			}
		default:
			start["text"] = ""
			for _, part := range halves(rep.text()) {
				deltas = append(deltas, map[string]any{"type": "text_delta", "text": part})
			}
		}
		s.event("content_block_start", map[string]any{"type": "content_block_start", "index": i, "content_block": start})
		for _, d := range deltas {
			s.event("content_block_delta", map[string]any{"type": "content_block_delta", "index": i, "delta": d})
		}
		s.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": i})
	}
	s.event("message_delta", map[string]any{"type": "message_delta",
		"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil}, "usage": map[string]any{"output_tokens": outputTokens}})
	s.event("message_stop", map[string]any{"type": "message_stop"})
}

func bedrock(w http.ResponseWriter, rep reply, stream bool) {
	stop := "end_turn"
	var input any
	json.Unmarshal([]byte(toolArgs), &input)
	if rep.tool == "json_tool_call" {
		json.Unmarshal([]byte(jsonAnswer), &input)
	}
	usage := map[string]any{"inputTokens": inputTokens, "outputTokens": outputTokens, "totalTokens": inputTokens + outputTokens,
		"cacheReadInputTokens": cacheRead, "cacheWriteInputTokens": cacheWrite}
	if rep.kind == "tool" {
		stop = "tool_use"
	}
	if !stream {
		var content []any
		if rep.kind == "thinking" {
			content = append(content, map[string]any{"reasoningContent": map[string]any{
				"reasoningText": map[string]any{"text": reasoningText, "signature": signature}}})
		}
		if rep.kind == "tool" {
			content = append(content, map[string]any{"toolUse": map[string]any{"toolUseId": toolID, "name": rep.tool, "input": input}})
		} else {
			content = append(content, map[string]any{"text": rep.text()})
		}
		writeJSON(w, map[string]any{"output": map[string]any{"message": map[string]any{"role": "assistant", "content": content}},
			"stopReason": stop, "usage": usage, "metrics": map[string]any{"latencyMs": 1}})
		return
	}
	w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
	w.WriteHeader(http.StatusOK)
	frame := func(event string, payload any) {
		b, _ := json.Marshal(payload)
		w.Write(eventStreamFrame(event, b))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}
	frame("messageStart", map[string]any{"role": "assistant"})
	index := 0
	if rep.kind == "thinking" {
		for _, part := range halves(reasoningText) {
			frame("contentBlockDelta", map[string]any{"contentBlockIndex": index,
				"delta": map[string]any{"reasoningContent": map[string]any{"text": part}}})
		}
		frame("contentBlockDelta", map[string]any{"contentBlockIndex": index,
			"delta": map[string]any{"reasoningContent": map[string]any{"signature": signature}}})
		frame("contentBlockStop", map[string]any{"contentBlockIndex": index})
		index++
	}
	if rep.kind == "tool" {
		frame("contentBlockStart", map[string]any{"contentBlockIndex": index,
			"start": map[string]any{"toolUse": map[string]any{"toolUseId": toolID, "name": rep.tool}}})
		args, _ := json.Marshal(input)
		for _, part := range halves(string(args)) {
			frame("contentBlockDelta", map[string]any{"contentBlockIndex": index,
				"delta": map[string]any{"toolUse": map[string]any{"input": part}}})
		}
	} else {
		for _, part := range halves(rep.text()) {
			frame("contentBlockDelta", map[string]any{"contentBlockIndex": index, "delta": map[string]any{"text": part}})
		}
	}
	frame("contentBlockStop", map[string]any{"contentBlockIndex": index})
	frame("messageStop", map[string]any{"stopReason": stop})
	frame("metadata", map[string]any{"usage": usage, "metrics": map[string]any{"latencyMs": 1}})
}

func gemini(w http.ResponseWriter, rep reply, stream bool) {
	var parts []any
	if rep.kind == "thinking" {
		parts = append(parts, map[string]any{"text": reasoningText, "thought": true})
	}
	if rep.kind == "tool" {
		var args any
		json.Unmarshal([]byte(toolArgs), &args)
		parts = append(parts, map[string]any{"functionCall": map[string]any{"name": rep.tool, "args": args}})
	} else {
		parts = append(parts, map[string]any{"text": rep.text()})
	}
	usage := map[string]any{"promptTokenCount": inputTokens, "candidatesTokenCount": outputTokens,
		"totalTokenCount": inputTokens + outputTokens, "cachedContentTokenCount": cacheRead}
	if rep.kind == "thinking" {
		usage["thoughtsTokenCount"] = reasoningCount
		usage["totalTokenCount"] = inputTokens + outputTokens + reasoningCount
	}
	candidate := func(parts []any, finish any) map[string]any {
		c := map[string]any{"content": map[string]any{"role": "model", "parts": parts}, "index": 0}
		if finish != nil {
			c["finishReason"] = finish
		}
		return c
	}
	if !stream {
		writeJSON(w, map[string]any{"candidates": []any{candidate(parts, "STOP")}, "usageMetadata": usage,
			"modelVersion": "gemini-parity", "responseId": "resp_parity"})
		return
	}
	s := newSSE(w, "text/event-stream")
	for i, part := range parts {
		var finish any
		if i == len(parts)-1 {
			finish = "STOP"
		}
		event := map[string]any{"candidates": []any{candidate([]any{part}, finish)}, "modelVersion": "gemini-parity",
			"responseId": "resp_parity"}
		if finish != nil {
			event["usageMetadata"] = usage
		}
		s.data(event)
	}
}

type sse struct{ w http.ResponseWriter }

func newSSE(w http.ResponseWriter, contentType string) sse {
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(http.StatusOK)
	return sse{w}
}

func (s sse) raw(text string) {
	io.WriteString(s.w, text)
	if f, ok := s.w.(http.Flusher); ok {
		f.Flush()
	}
}

func (s sse) data(v any) {
	b, _ := json.Marshal(v)
	s.raw("data: " + string(b) + "\n\n")
}

func (s sse) event(name string, v any) {
	b, _ := json.Marshal(v)
	s.raw("event: " + name + "\ndata: " + string(b) + "\n\n")
}

// eventStreamFrame encodes one AWS event-stream message, as mockllm does.
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

// halves splits text in two, so every stream carries more than one delta per block.
func halves(text string) []string {
	mid := len(text) / 2
	return []string{text[:mid], text[mid:]}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// dig walks maps by key and slices by index, returning nil when anything is missing.
func dig(v any, path ...any) any {
	for _, step := range path {
		switch s := step.(type) {
		case string:
			m, ok := v.(map[string]any)
			if !ok {
				return nil
			}
			v = m[s]
		case int:
			a, ok := v.([]any)
			if !ok || s >= len(a) {
				return nil
			}
			v = a[s]
		}
	}
	return v
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
