package main

import (
	"encoding/json"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDecisionsBodyCarriesTheQuestionsAndTheDirective(t *testing.T) {
	pool := newTextPool(1<<16, 1)
	res := result{Model: "gpt-6-luna-0", CtxTokens: 2000, TTFTPlanMs: 0, DurPlanMs: 80}
	body := buildDecisionsBody(pool, rand.New(rand.NewPCG(1, 2)), res, 20, 5, "0-1")

	var parsed struct {
		Model     string `json:"model"`
		Input     string `json:"input"`
		Questions []struct {
			Type string `json:"type"`
			Name string `json:"name"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("%v: %s", err, body)
	}
	if parsed.Model != "gpt-6-luna-0" || len(parsed.Questions) != 5 {
		t.Fatalf("model %q with %d questions", parsed.Model, len(parsed.Questions))
	}
	types := []string{"predicate", "choice", "score", "predicate", "choice"}
	for i, q := range parsed.Questions {
		if q.Type != types[i] || q.Name != "q"+string(rune('0'+i)) {
			t.Fatalf("question %d is %+v", i, q)
		}
	}
	if !strings.HasPrefix(parsed.Input, "case 0-1 ") || !strings.HasSuffix(parsed.Input, " [[mock ttft_ms=0 dur_ms=80 out=20 in=2000]]") {
		t.Fatalf("input %q", parsed.Input[len(parsed.Input)-80:])
	}
	if len(parsed.Input) < 2000*bytesPerToken {
		t.Fatalf("input is %d bytes", len(parsed.Input))
	}
}

func TestEveryWorkloadLoads(t *testing.T) {
	paths, err := filepath.Glob("../../workloads/*.json")
	if err != nil || len(paths) == 0 {
		t.Fatalf("no workloads: %v", err)
	}
	for _, path := range paths {
		if _, err := loadWorkload(path); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
	}
}

func TestDecisionsClassesMayNotStream(t *testing.T) {
	path := filepath.Join(t.TempDir(), "w.json")
	workload := `{"stages":[{"duration_s":1,"rps":1}],"models":{"decisions":{"prefix":"m-","count":1}},
		"classes":[{"name":"d","weight":1,"endpoint":"decisions","stream_fraction":0.5}]}`
	if err := os.WriteFile(path, []byte(workload), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadWorkload(path); err == nil || !strings.Contains(err.Error(), "decisions do not stream") {
		t.Fatalf("loadWorkload = %v", err)
	}
}
