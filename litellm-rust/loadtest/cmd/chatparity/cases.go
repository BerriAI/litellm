package main

import (
	"fmt"
	"strings"
)

// deployment is one model group in the shared config, pointing at the mock.
type deployment struct {
	name   string         // model group name the client asks for
	family string         // which scenarios apply: openai, claude, gemini or compat
	params map[string]any // litellm_params, with {mock} replaced by the mock's URL
}

func deployments(credentials string) []deployment {
	compat := func(name, model string) deployment {
		return deployment{name, "compat", map[string]any{"model": model, "api_base": "{mock}/v1", "api_key": "sk-parity-upstream"}}
	}
	vertex := map[string]any{"api_base": "{mock}", "vertex_project": "parity-project", "vertex_credentials": credentials}
	with := func(base map[string]any, extra map[string]any) map[string]any {
		out := map[string]any{}
		for k, v := range base {
			out[k] = v
		}
		for k, v := range extra {
			out[k] = v
		}
		return out
	}
	return []deployment{
		{"openai", "openai", map[string]any{"model": "openai/gpt-4o", "api_base": "{mock}/v1", "api_key": "sk-parity-upstream"}},
		{"azure", "openai", map[string]any{"model": "azure/gpt-4o", "api_base": "{mock}", "api_key": "sk-parity-upstream",
			"api_version": "2025-02-01-preview"}},
		{"anthropic", "claude", map[string]any{"model": "anthropic/claude-sonnet-4-5-20250929", "api_base": "{mock}",
			"api_key": "sk-parity-upstream"}},
		{"bedrock", "claude", map[string]any{"model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
			"aws_access_key_id": "AKIAPARITYPARITY", "aws_secret_access_key": "parity-secret", "aws_region_name": "us-east-1",
			"aws_bedrock_runtime_endpoint": "{mock}"}},
		// The same model with the endpoint as api_base and credentials from the environment
		// (chat-parity.sh sets them), for a host that does not read AWS settings from the config.
		{"bedrock-env", "claude", map[string]any{"model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "api_base": "{mock}"}},
		{"vertex-claude", "claude", with(vertex, map[string]any{"model": "vertex_ai/claude-sonnet-4-5@20250929",
			"vertex_location": "us-east5"})},
		{"gemini", "gemini", map[string]any{"model": "gemini/gemini-2.5-flash", "api_base": "{mock}", "api_key": "parity-google-key"}},
		{"vertex-gemini", "gemini", with(vertex, map[string]any{"model": "vertex_ai/gemini-2.5-flash", "vertex_location": "us-central1"})},
		compat("groq", "groq/llama-3.3-70b-versatile"),
		compat("mistral", "mistral/mistral-large-latest"),
		compat("deepseek", "deepseek/deepseek-chat"),
		compat("together", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"),
		compat("fireworks", "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct"),
		compat("xai", "xai/grok-3"),
		compat("openrouter", "openrouter/openai/gpt-4o"),
		compat("cerebras", "cerebras/llama3.1-8b"),
		compat("perplexity", "perplexity/sonar"),
		compat("hosted-vllm", "hosted_vllm/Qwen/Qwen3-8B"),
		compat("nvidia-nim", "nvidia_nim/meta/llama-3.3-70b-instruct"),
	}
}

// scenario is one shape of chat request, sent to every deployment whose family it lists.
type scenario struct {
	name     string
	families string // space-separated families, or "all"
	streams  []bool // which of non-streaming and streaming to send
	body     func(mock string) map[string]any
}

var both = []bool{false, true}

func user(text string) map[string]any { return map[string]any{"role": "user", "content": text} }

var weatherTool = map[string]any{"type": "function", "function": map[string]any{
	"name": "get_weather", "description": "Current weather for a city",
	"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required": []any{"city"}},
}}

const pngDataURL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

var scenarios = []scenario{
	{"basic", "all", both, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Say hello.")}, "max_tokens": 50, "temperature": 0.2}
	}},
	{"include-usage", "all", []bool{true}, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Say hello.")}, "max_tokens": 50,
			"stream_options": map[string]any{"include_usage": true}}
	}},
	{"system-multi-turn", "all", both, func(string) map[string]any {
		return map[string]any{"messages": []any{
			map[string]any{"role": "system", "content": "You are terse."},
			user("Hi."),
			map[string]any{"role": "assistant", "content": "Hello."},
			user("Say hello again."),
		}, "max_tokens": 50, "top_p": 0.9, "stop": []any{"END"}, "user": "parity-user"}
	}},
	{"tools", "all", both, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Weather in Paris? [[parity reply=tool]]")},
			"tools": []any{weatherTool}, "tool_choice": "auto", "max_tokens": 100}
	}},
	{"tools-include-usage", "all", []bool{true}, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Weather in Paris? [[parity reply=tool]]")},
			"tools": []any{weatherTool}, "tool_choice": "required", "max_tokens": 100,
			"stream_options": map[string]any{"include_usage": true}}
	}},
	{"tool-result", "all", both, func(string) map[string]any {
		return map[string]any{"messages": []any{
			user("Weather in Paris?"),
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{"id": "call_prev_1",
				"type": "function", "function": map[string]any{"name": "get_weather", "arguments": `{"city": "Paris"}`}}}},
			map[string]any{"role": "tool", "tool_call_id": "call_prev_1", "content": "18C and sunny"},
		}, "tools": []any{weatherTool}, "max_tokens": 100}
	}},
	{"json-schema", "all", both, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Answer as JSON. [[parity reply=json]]")}, "max_tokens": 100,
			"response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "answer", "strict": true,
				"schema": map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "integer"}},
					"required": []any{"answer"}, "additionalProperties": false}}}}
	}},
	{"json-object", "openai gemini compat", []bool{false}, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Answer as JSON. [[parity reply=json]]")}, "max_tokens": 100,
			"response_format": map[string]any{"type": "json_object"}}
	}},
	{"image-data-url", "openai claude gemini", both, func(string) map[string]any {
		return map[string]any{"messages": []any{map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "What is in this image?"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": pngDataURL}},
		}}}, "max_tokens": 50}
	}},
	{"image-remote-url", "openai claude gemini", []bool{false}, func(mock string) map[string]any {
		return map[string]any{"messages": []any{map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "What is in this image?"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": mock + "/img.png"}},
		}}}, "max_tokens": 50}
	}},
	{"reasoning-effort", "claude gemini", both, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Think, then say hello. [[parity reply=thinking]]")},
			"reasoning_effort": "low", "max_tokens": 4096}
	}},
	{"cache-control", "claude", []bool{false}, func(string) map[string]any {
		return map[string]any{"messages": []any{
			map[string]any{"role": "system", "content": []any{map[string]any{"type": "text", "text": "Long shared prefix.",
				"cache_control": map[string]any{"type": "ephemeral"}}}},
			user("Say hello."),
		}, "max_tokens": 50}
	}},
	{"unsupported-param", "all", []bool{false}, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Say hello.")}, "max_tokens": 50, "n": 2, "logit_bias": map[string]any{"50256": -100}}
	}},
	{"drop-params", "all", []bool{false}, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Say hello.")}, "max_tokens": 50, "n": 2,
			"logit_bias": map[string]any{"50256": -100}, "drop_params": true}
	}},
	{"max-completion-tokens", "all", []bool{false}, func(string) map[string]any {
		return map[string]any{"messages": []any{user("Say hello.")}, "max_completion_tokens": 60, "seed": 7}
	}},
}

// testCase is one request, sent once to each proxy.
type testCase struct {
	Name   string         `json:"name"`
	Model  string         `json:"model"`
	Stream bool           `json:"stream"`
	Body   map[string]any `json:"body"`
}

func buildCases(deps []deployment, mockURL, filter string) []testCase {
	var cases []testCase
	for _, d := range deps {
		for _, s := range scenarios {
			if s.families != "all" && !strings.Contains(" "+s.families+" ", " "+d.family+" ") {
				continue
			}
			for _, stream := range s.streams {
				body := s.body(mockURL)
				body["model"] = d.name
				mode := "unary"
				if stream {
					body["stream"] = true
					mode = "stream"
				}
				name := fmt.Sprintf("%s/%s/%s", d.name, s.name, mode)
				if filter != "" && !strings.Contains(name, filter) {
					continue
				}
				cases = append(cases, testCase{name, d.name, stream, body})
			}
		}
	}
	return cases
}
