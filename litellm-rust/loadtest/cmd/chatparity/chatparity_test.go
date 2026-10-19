package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func decode(t *testing.T, text string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(text), &v); err != nil {
		t.Fatal(err)
	}
	return v
}

func TestDiffValuesReportsMissingAndChangedPaths(t *testing.T) {
	var diffs []difference
	diffValues("response", "", decode(t, `{"a": 1, "b": [1, 2], "c": {"d": "x"}}`),
		decode(t, `{"a": 1, "b": [1], "c": {"d": "y"}, "e": true}`), &diffs)
	got := map[string]bool{}
	for _, d := range diffs {
		got[d.Path] = true
	}
	for _, want := range []string{".b[1]", ".c.d", ".e"} {
		if !got[want] {
			t.Errorf("missing %s in %+v", want, diffs)
		}
	}
	if len(diffs) != 3 {
		t.Errorf("want 3 differences, got %+v", diffs)
	}
}

func TestNormalizeIDsKeepsMockIDsAndPrefixesOfGeneratedOnes(t *testing.T) {
	got := normalizeIDs(decode(t, `{"id": "chatcmpl-123", "created": 5, "choices": [{"message": {"tool_calls": [{"id": "call_parity_1"}, {"id": "call_abc"}]}}]}`))
	want := decode(t, `{"id": "chatcmpl-*", "created": "<timestamp>", "choices": [{"message": {"tool_calls": [{"id": "call_parity_1"}, {"id": "call_*"}]}}]}`)
	if !equal(got, want) {
		t.Errorf("got %v", got)
	}
}

func TestAssembleFoldsContentToolCallsAndUsage(t *testing.T) {
	s := parseSSE([]byte(strings.Join([]string{
		`data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		`data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_parity_1","type":"function","function":{"name":"f","arguments":"{\"a\""}}]}}]}`,
		`data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":": 1}"}}]},"finish_reason":"tool_calls"}]}`,
		`data: {"id":"chatcmpl-1","model":"m","choices":[],"usage":{"total_tokens":3}}`,
		`data: [DONE]`,
	}, "\n\n")))
	got := assemble(s)
	if got["content"] != "Hello" || got["done"] != true || got["one_id_per_stream"] != true {
		t.Errorf("got %v", got)
	}
	calls := got["tool_calls"].([]any)
	if len(calls) != 1 || calls[0].(map[string]any)["arguments"] != `{"a": 1}` {
		t.Errorf("tool calls %v", calls)
	}
	if !equal(got["finish_reasons"], []any{"tool_calls"}) || !equal(got["usage"], map[string]any{"total_tokens": 3.0}) {
		t.Errorf("got %v", got)
	}
}

func TestNormalizeUpstreamDropsTransportHeadersAndTheSignature(t *testing.T) {
	got := normalizeUpstream([]upstream{{Method: "POST", Path: "/p", Headers: map[string]string{
		"user-agent":        "x",
		"x-amz-date":        "20261019T000000Z",
		"authorization":     "AWS4-HMAC-SHA256 Credential=AKIA/20261019/us-east-1/bedrock/aws4_request, SignedHeaders=host, Signature=abc",
		"anthropic-version": "2023-06-01",
	}}})
	headers := got.([]any)[0].(map[string]any)["headers"]
	want := map[string]any{"authorization": "AWS4-HMAC-SHA256 region=us-east-1 service=bedrock", "anthropic-version": "2023-06-01"}
	if !equal(headers, want) {
		t.Errorf("got %v", headers)
	}
}

func TestMarkKnownMatchesCaseWhereAndPathGlobs(t *testing.T) {
	diffs := []difference{{Where: "upstream", Path: "[0].headers.accept"}, {Where: "response", Path: ".body.model"}}
	markKnown("openai/basic/unary", diffs, []known{{Case: "*/basic/*", Where: "upstream", Path: "*.accept", Reason: "r"}})
	if diffs[0].Known != "r" || diffs[1].Known != "" {
		t.Errorf("got %+v", diffs)
	}
}

func TestMockAnswersAnthropicJSONModeThroughTheForcedTool(t *testing.T) {
	m := &mock{}
	body := `{"model":"c","tools":[{"name":"json_tool_call"}],"messages":[{"role":"user","content":"x"}]}`
	w := httptest.NewRecorder()
	m.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
	got := decode(t, w.Body.String())
	if dig(got, "content", 0, "name") != "json_tool_call" || dig(got, "content", 0, "input", "answer") != 42.0 {
		t.Errorf("got %v", got)
	}
	if recorded := m.take(); len(recorded) != 1 || recorded[0].Path != "/v1/messages" {
		t.Errorf("recorded %+v", recorded)
	}
}

func TestEveryCaseNameIsUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, c := range buildCases(deployments(""), "http://mock", "") {
		if seen[c.Name] {
			t.Errorf("duplicate case %s", c.Name)
		}
		seen[c.Name] = true
	}
}
