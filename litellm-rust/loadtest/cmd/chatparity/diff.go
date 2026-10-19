package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// difference is one place where the two proxies disagree. Where is "upstream" (what the
// provider was sent), "response" (what the client got, or the assembled stream) or "chunks"
// (the stream chunk by chunk, informational: boundaries may differ for the same content).
type difference struct {
	Where  string `json:"where"`
	Path   string `json:"path"`
	Python any    `json:"python"`
	Rust   any    `json:"rust"`
	Known  string `json:"known,omitempty"` // the reason, when the difference is listed as known
}

const absent = "<absent>"

// diffValues appends every path where a and b differ.
func diffValues(where, path string, a, b any, out *[]difference) {
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok {
			break
		}
		keys := map[string]bool{}
		for k := range av {
			keys[k] = true
		}
		for k := range bv {
			keys[k] = true
		}
		for _, k := range sortedKeys(keys) {
			x, xok := av[k]
			y, yok := bv[k]
			switch {
			case !xok:
				*out = append(*out, difference{Where: where, Path: path + "." + k, Python: absent, Rust: y})
			case !yok:
				*out = append(*out, difference{Where: where, Path: path + "." + k, Python: x, Rust: absent})
			default:
				diffValues(where, path+"."+k, x, y, out)
			}
		}
		return
	case []any:
		bv, ok := b.([]any)
		if !ok {
			break
		}
		for i := 0; i < len(av) || i < len(bv); i++ {
			p := fmt.Sprintf("%s[%d]", path, i)
			switch {
			case i >= len(av):
				*out = append(*out, difference{Where: where, Path: p, Python: absent, Rust: bv[i]})
			case i >= len(bv):
				*out = append(*out, difference{Where: where, Path: p, Python: av[i], Rust: absent})
			default:
				diffValues(where, p, av[i], bv[i], out)
			}
		}
		return
	}
	if !equal(a, b) {
		*out = append(*out, difference{Where: where, Path: path, Python: a, Rust: b})
	}
}

func equal(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return bytes.Equal(x, y)
}

// Headers that differ for reasons unrelated to translation: client libraries, transport,
// tracing, and request signing that depends on the clock.
var ignoredHeaders = regexp.MustCompile(`^(user-agent|content-length|accept-encoding|connection|host|traceparent|tracestate|` +
	`x-stainless-.*|amz-sdk-.*|x-amzn-trace-id|x-amz-date|x-amz-content-sha256|x-goog-api-client|x-litellm-.*)$`)

var sigV4 = regexp.MustCompile(`^AWS4-HMAC-SHA256 Credential=[^/]+/\d{8}/([^/]+)/([^/]+)/aws4_request`)

func normalizeUpstream(requests []upstream) any {
	out := make([]any, len(requests))
	for i, r := range requests {
		headers := map[string]any{}
		for name, value := range r.Headers {
			if ignoredHeaders.MatchString(name) {
				continue
			}
			if name == "authorization" {
				if m := sigV4.FindStringSubmatch(value); m != nil {
					value = fmt.Sprintf("AWS4-HMAC-SHA256 region=%s service=%s", m[1], m[2])
				}
			}
			headers[name] = value
		}
		entry := map[string]any{"method": r.Method, "path": r.Path, "headers": headers, "body": r.Body}
		if r.Query != "" {
			entry["query"] = r.Query
		}
		out[i] = entry
	}
	return out
}

// normalizeIDs replaces ids and timestamps the proxies generate with a placeholder that keeps
// their prefix (`chatcmpl-*`, `call_*`), so only their shape is compared. Ids the mock chose
// contain "parity" and are compared exactly.
func normalizeIDs(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := map[string]any{}
		for k, x := range t {
			switch {
			case k == "created":
				out[k] = "<timestamp>"
			case k == "id":
				out[k] = generated(x)
			default:
				out[k] = normalizeIDs(x)
			}
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, x := range t {
			out[i] = normalizeIDs(x)
		}
		return out
	}
	return v
}

func generated(v any) any {
	s, ok := v.(string)
	if !ok || strings.Contains(s, "parity") {
		return v
	}
	if i := strings.IndexAny(s, "-_"); i >= 0 {
		return s[:i+1] + "*"
	}
	return "*"
}

// stream is a parsed SSE response.
type stream struct {
	Chunks []any `json:"chunks"`
	Done   bool  `json:"done"`
	Other  []any `json:"other,omitempty"` // lines that are neither JSON data nor [DONE]
}

func parseSSE(body []byte) stream {
	var s stream
	scanner := bufio.NewScanner(bytes.NewReader(body))
	scanner.Buffer(make([]byte, 1<<20), 16<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			s.Other = append(s.Other, line)
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			s.Done = true
			continue
		}
		var chunk any
		if json.Unmarshal([]byte(data), &chunk) != nil {
			s.Other = append(s.Other, line)
			continue
		}
		s.Chunks = append(s.Chunks, chunk)
	}
	return s
}

// assemble folds a stream's chunks into what a client library would build from them, which is
// what must agree; how the content was split into chunks may differ.
func assemble(s stream) map[string]any {
	var content, reasoning strings.Builder
	type call struct{ id, kind, name, args any }
	calls := map[float64]*call{}
	var order []float64
	ids := map[string]bool{}
	models := map[string]bool{}
	var roles, finishes []any
	var usage, errorFrame any
	extra := map[string]bool{}
	sawContent, sawReasoning := false, false
	for _, c := range s.Chunks {
		chunk, _ := c.(map[string]any)
		if e, ok := chunk["error"]; ok {
			errorFrame = e
			continue
		}
		if id, ok := chunk["id"].(string); ok {
			ids[id] = true
		}
		if m, ok := chunk["model"].(string); ok {
			models[m] = true
		}
		if u, ok := chunk["usage"]; ok && u != nil {
			usage = u
		}
		for k := range chunk {
			switch k {
			case "id", "object", "created", "model", "choices", "usage", "system_fingerprint":
			default:
				extra[k] = true
			}
		}
		choices, _ := chunk["choices"].([]any)
		for _, ch := range choices {
			choice, _ := ch.(map[string]any)
			if f := choice["finish_reason"]; f != nil {
				finishes = append(finishes, f)
			}
			delta, _ := choice["delta"].(map[string]any)
			for k, v := range delta {
				switch k {
				case "role":
					if v != nil {
						roles = append(roles, v)
					}
				case "content":
					if t, ok := v.(string); ok {
						content.WriteString(t)
						sawContent = true
					}
				case "reasoning_content":
					if t, ok := v.(string); ok {
						reasoning.WriteString(t)
						sawReasoning = true
					}
				case "tool_calls":
					list, _ := v.([]any)
					for _, item := range list {
						tc, _ := item.(map[string]any)
						index, _ := tc["index"].(float64)
						entry := calls[index]
						if entry == nil {
							entry = &call{}
							calls[index] = entry
							order = append(order, index)
						}
						if x := tc["id"]; x != nil {
							entry.id = generated(x)
						}
						if x := tc["type"]; x != nil {
							entry.kind = x
						}
						fn, _ := tc["function"].(map[string]any)
						if x := fn["name"]; x != nil {
							entry.name = x
						}
						if x, ok := fn["arguments"].(string); ok {
							prev, _ := entry.args.(string)
							entry.args = prev + x
						}
					}
				default:
					if v != nil {
						extra["delta."+k] = true
					}
				}
			}
		}
	}
	sort.Float64s(order)
	var toolCalls []any
	for _, i := range order {
		c := calls[i]
		toolCalls = append(toolCalls, map[string]any{"index": i, "id": c.id, "type": c.kind, "name": c.name, "arguments": c.args})
	}
	out := map[string]any{
		"one_id_per_stream": len(ids) <= 1,
		"id_prefix":         nil,
		"models":            sortedKeys(models),
		"roles":             roles,
		"finish_reasons":    finishes,
		"usage":             usage,
		"done":              s.Done,
		"tool_calls":        toolCalls,
		"other_fields":      sortedKeys(extra),
		"non_data_lines":    s.Other,
	}
	for id := range ids {
		out["id_prefix"] = generated(id)
	}
	if sawContent {
		out["content"] = content.String()
	}
	if sawReasoning {
		out["reasoning_content"] = reasoning.String()
	}
	if errorFrame != nil {
		out["error"] = errorFrame
	}
	return out
}

// known is a difference the team has looked at and accepted, with the reason.
type known struct {
	Case   string `json:"case"`  // glob over case names; * matches anything
	Where  string `json:"where"` // upstream, response or chunks; empty matches all
	Path   string `json:"path"`  // glob over the difference's path
	Reason string `json:"reason"`
}

func glob(pattern string) *regexp.Regexp {
	parts := strings.Split(pattern, "*")
	for i, p := range parts {
		parts[i] = regexp.QuoteMeta(p)
	}
	return regexp.MustCompile("^" + strings.Join(parts, ".*") + "$")
}

func markKnown(name string, diffs []difference, list []known) {
	for i := range diffs {
		for _, k := range list {
			if (k.Where == "" || k.Where == diffs[i].Where) && glob(k.Case).MatchString(name) && glob(k.Path).MatchString(diffs[i].Path) {
				diffs[i].Known = k.Reason
				break
			}
		}
	}
}
