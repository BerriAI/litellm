package main

import "testing"

func TestRequestModelReadsTheTopLevelModel(t *testing.T) {
	for body, want := range map[string]string{
		`{"model": "gpt-6-astra", "messages":[{"content":"say \"model\": \"x\""}]}`: "gpt-6-astra",
		`{"messages":[],"model":"claude-opus-5-5"}`:                                 "claude-opus-5-5",
		`{"messages":[]}`: "mock",
	} {
		if got := requestModel([]byte(body)); got != want {
			t.Fatalf("requestModel(%s) = %q, want %q", body, got, want)
		}
	}
}

func TestDeploymentOfReadsBedrockModelPaths(t *testing.T) {
	for path, want := range map[string][2]string{
		"/d/chat-0-1/v1/chat/completions":                          {"chat-0-1", "/v1/chat/completions"},
		"/model/d/bedrock-0-0/us.anthropic.claude/converse-stream": {"bedrock-0-0", "/model/us.anthropic.claude/converse-stream"},
		"/v1/messages": {"default", "/v1/messages"},
	} {
		if id, rest := deploymentOf(path); id != want[0] || rest != want[1] {
			t.Fatalf("deploymentOf(%s) = %s, %s", path, id, rest)
		}
	}
}
