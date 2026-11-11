package main

import (
	"encoding/json"
	"testing"
)

func TestDecisionAnswersFollowTheQuestions(t *testing.T) {
	body := `{"model":"gpt-6-luna","input":"x [[mock out=20]]","questions":[` +
		`{"type":"predicate","name":"p","instructions":"q"},` +
		`{"type":"choice","name":"c","instructions":"q","choices":[{"value":"low"},{"value":true}]},` +
		`{"type":"score","name":"s","instructions":"q","levels":[{"label":"calm"},{"label":"angry"}]},` +
		`{"type":"box","name":"b","instructions":"q"}]}`
	answers, err := decisionAnswers([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(answers)
	want := `[{"name":"p","probability":0.9,"type":"predicate"},` +
		`{"choice":"low","confidence":0.9,"name":"c","probabilities":[{"probability":0.9,"value":"low"},{"probability":0.1,"value":true}],"type":"choice"},` +
		`{"confidence":0.9,"name":"s","probabilities":[{"label":"calm","probability":0.1,"value":0},{"label":"angry","probability":0.9,"value":1}],"score":1,"type":"score"},` +
		`{"box":{"bottom":0.9,"left":0.1,"right":0.9,"top":0.1},"confidence":0.9,"name":"b","type":"box"}]`
	if string(got) != want {
		t.Fatalf("decisionAnswers = %s\nwant %s", got, want)
	}
}

func TestDecisionAnswersRefuseABodyWithoutQuestions(t *testing.T) {
	for _, body := range []string{`{"input":"x"}`, `{"questions":[{"type":"choice","name":"c"}]}`, `not json`} {
		if _, err := decisionAnswers([]byte(body)); err == nil {
			t.Fatalf("decisionAnswers(%s) succeeded", body)
		}
	}
}

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
