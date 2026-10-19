// chatparity sends the same chat completions requests through the Python proxy and the Rust
// gateway, both pointed at one recording mock provider, and reports every difference in what the
// provider was sent and what the client got back.
//
//	chatparity config -mock http://127.0.0.1:9100 -out out   # config.yaml both proxies load
//	chatparity run -mock-addr 127.0.0.1:9100 -python http://127.0.0.1:4001 \
//	    -rust http://127.0.0.1:4002 -out out [-known known.json] [-filter anthropic/]
//
// The mock answers each provider's API (OpenAI-compatible, Azure, Anthropic, Bedrock Converse,
// Gemini, Vertex) the same way for both proxies, so differences in the client's answer come from
// the proxies' translation. scripts/chat-parity.sh builds and starts everything locally.
package main

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const masterKey = "sk-parity-master"

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		log.Fatal("usage: chatparity config|run ...")
	}
	switch os.Args[1] {
	case "config":
		configCmd(os.Args[2:])
	case "run":
		runCmd(os.Args[2:])
	default:
		log.Fatalf("unknown subcommand %q", os.Args[1])
	}
}

// configCmd writes the one config both proxies load: every deployment pointed at the mock, and
// a throwaway service account whose token endpoint is the mock, for the Vertex deployments.
func configCmd(args []string) {
	fs := flag.NewFlagSet("config", flag.ExitOnError)
	mockURL := fs.String("mock", "http://127.0.0.1:9100", "URL the proxies reach the mock at")
	out := fs.String("out", "out", "directory to write config.yaml to")
	fs.Parse(args)
	must(os.MkdirAll(*out, 0o755))
	var models []any
	for _, d := range deployments(serviceAccount(*mockURL)) {
		params := map[string]any{}
		for k, v := range d.params {
			if s, ok := v.(string); ok {
				v = strings.ReplaceAll(s, "{mock}", *mockURL)
			}
			params[k] = v
		}
		models = append(models, map[string]any{"model_name": d.name, "litellm_params": params})
	}
	// JSON is YAML, and both proxies read it as their config.yaml.
	config := map[string]any{
		"model_list":       models,
		"general_settings": map[string]any{"master_key": masterKey},
		"litellm_settings": map[string]any{"num_retries": 0, "user_url_allowed_hosts": []any{mockHost(*mockURL)}},
	}
	b, _ := json.MarshalIndent(config, "", "  ")
	must(os.WriteFile(filepath.Join(*out, "config.yaml"), b, 0o644))
	fmt.Println(filepath.Join(*out, "config.yaml"))
}

// mockHost is the mock's host:port, allowed as a media URL host so remote image cases reach it.
func mockHost(mockURL string) string {
	u, err := url.Parse(mockURL)
	must(err)
	return u.Host
}

func serviceAccount(mockURL string) string {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	must(err)
	der, err := x509.MarshalPKCS8PrivateKey(key)
	must(err)
	b, _ := json.Marshal(map[string]any{
		"type": "service_account", "project_id": "parity-project", "private_key_id": "parity",
		"private_key":  string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"client_email": "parity@parity-project.iam.gserviceaccount.com", "client_id": "1",
		"token_uri": mockURL + "/token",
	})
	return string(b)
}

// exchange is one side's request and what came of it.
type exchange struct {
	Status      int        `json:"status"`
	ContentType string     `json:"content_type"`
	Body        any        `json:"body"`
	Upstream    []upstream `json:"upstream"`
	Error       string     `json:"error,omitempty"`
}

type outcome struct {
	Case        string       `json:"case"`
	Python      exchange     `json:"python"`
	Rust        exchange     `json:"rust"`
	Differences []difference `json:"differences"`
}

func runCmd(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	mockAddr := fs.String("mock-addr", "127.0.0.1:9100", "address the mock listens on")
	mockURL := fs.String("mock", "", "URL the proxies reach the mock at (default http://<mock-addr>)")
	python := fs.String("python", "http://127.0.0.1:4001", "Python proxy base URL")
	rust := fs.String("rust", "http://127.0.0.1:4002", "Rust gateway base URL")
	out := fs.String("out", "out", "directory for report.md, results.json and per-case captures")
	knownPath := fs.String("known", "", "JSON list of accepted differences")
	filter := fs.String("filter", "", "only run cases whose name contains this")
	fs.Parse(args)
	if *mockURL == "" {
		*mockURL = "http://" + *mockAddr
	}
	var accepted []known
	if *knownPath != "" {
		b, err := os.ReadFile(*knownPath)
		must(err)
		must(json.Unmarshal(b, &accepted))
	}

	m := &mock{}
	listener, err := net.Listen("tcp", *mockAddr)
	must(err)
	go http.Serve(listener, m)

	client := &http.Client{Timeout: 120 * time.Second}
	cases := buildCases(deployments(""), *mockURL, *filter)
	must(os.MkdirAll(filepath.Join(*out, "cases"), 0o755))
	var outcomes []outcome
	for _, c := range cases {
		o := outcome{Case: c.Name}
		o.Python = send(client, m, *python, c)
		o.Rust = send(client, m, *rust, c)
		o.Differences = compare(c, o.Python, o.Rust)
		markKnown(c.Name, o.Differences, accepted)
		outcomes = append(outcomes, o)
		b, _ := json.MarshalIndent(map[string]any{"request": c.Body, "outcome": o}, "", "  ")
		must(os.WriteFile(filepath.Join(*out, "cases", strings.ReplaceAll(c.Name, "/", "__")+".json"), b, 0o644))
		fmt.Printf("%-55s %s\n", c.Name, verdict(o.Differences))
	}
	b, _ := json.MarshalIndent(outcomes, "", "  ")
	must(os.WriteFile(filepath.Join(*out, "results.json"), b, 0o644))
	failing := writeReport(filepath.Join(*out, "report.md"), outcomes)
	fmt.Printf("\n%d cases, %d with unexplained differences; report in %s\n", len(outcomes), failing, filepath.Join(*out, "report.md"))
	if failing > 0 {
		os.Exit(1)
	}
}

func send(client *http.Client, m *mock, base string, c testCase) exchange {
	m.take()
	body, _ := json.Marshal(c.Body)
	req, _ := http.NewRequest(http.MethodPost, strings.TrimRight(base, "/")+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+masterKey)
	resp, err := client.Do(req)
	if err != nil {
		return exchange{Error: err.Error(), Upstream: m.take()}
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	x := exchange{Status: resp.StatusCode, ContentType: resp.Header.Get("Content-Type"), Upstream: m.take()}
	if strings.HasPrefix(x.ContentType, "text/event-stream") {
		x.Body = parseSSE(raw)
	} else {
		var parsed any
		if json.Unmarshal(raw, &parsed) == nil {
			x.Body = parsed
		} else {
			x.Body = string(raw)
		}
	}
	return x
}

func compare(c testCase, py, rs exchange) []difference {
	var diffs []difference
	diffValues("upstream", "", normalizeUpstream(py.Upstream), normalizeUpstream(rs.Upstream), &diffs)
	top := func(x exchange) map[string]any {
		return map[string]any{"status": x.Status, "content_type": mediaType(x.ContentType), "transport_error": x.Error}
	}
	diffValues("response", "", top(py), top(rs), &diffs)
	ps, pok := py.Body.(stream)
	rsm, rok := rs.Body.(stream)
	switch {
	case pok && rok:
		diffValues("response", ".stream", assemble(ps), assemble(rsm), &diffs)
		diffValues("chunks", "", normalizeIDs(ps.Chunks), normalizeIDs(rsm.Chunks), &diffs)
	case !pok && !rok:
		diffValues("response", ".body", normalizeIDs(py.Body), normalizeIDs(rs.Body), &diffs)
	}
	return diffs
}

func mediaType(contentType string) string {
	media, _, _ := strings.Cut(contentType, ";")
	return strings.TrimSpace(media)
}

func counts(diffs []difference) (open, accepted, chunks int) {
	for _, d := range diffs {
		switch {
		case d.Where == "chunks":
			chunks++
		case d.Known != "":
			accepted++
		default:
			open++
		}
	}
	return
}

func verdict(diffs []difference) string {
	open, accepted, chunks := counts(diffs)
	if open == 0 {
		return fmt.Sprintf("ok (%d known, %d chunk-level)", accepted, chunks)
	}
	return fmt.Sprintf("DIFF %d (%d known, %d chunk-level)", open, accepted, chunks)
}

// writeReport writes a summary table and every case's open differences, and returns how many
// cases have any.
func writeReport(path string, outcomes []outcome) int {
	var b strings.Builder
	failing := 0
	b.WriteString("# Chat completions parity: Python proxy vs Rust gateway\n\n")
	b.WriteString("Open differences fail the run. Known ones are listed in the known-differences file with a reason. ")
	b.WriteString("Chunk-level differences compare streams chunk by chunk and are informational: the assembled stream is what must agree.\n\n")
	b.WriteString("| case | python | rust | open | known | chunk-level |\n|---|---|---|---|---|---|\n")
	for _, o := range outcomes {
		open, accepted, chunks := counts(o.Differences)
		if open > 0 {
			failing++
		}
		fmt.Fprintf(&b, "| %s | %d | %d | %d | %d | %d |\n", o.Case, o.Python.Status, o.Rust.Status, open, accepted, chunks)
	}
	b.WriteString("\n## Open differences\n")
	for _, o := range outcomes {
		var lines []string
		for _, d := range o.Differences {
			if d.Where == "chunks" || d.Known != "" {
				continue
			}
			lines = append(lines, fmt.Sprintf("| %s | `%s` | %s | %s |", d.Where, d.Path, cell(d.Python), cell(d.Rust)))
		}
		if len(lines) == 0 {
			continue
		}
		fmt.Fprintf(&b, "\n### %s\n\n| where | path | python | rust |\n|---|---|---|---|\n%s\n", o.Case, strings.Join(lines, "\n"))
	}
	must(os.WriteFile(path, []byte(b.String()), 0o644))
	return failing
}

func cell(v any) string {
	s, ok := v.(string)
	if !ok {
		raw, _ := json.Marshal(v)
		s = string(raw)
	}
	if len(s) > 300 {
		s = s[:300] + "…"
	}
	return "`" + strings.NewReplacer("|", "\\|", "\n", " ", "`", "'").Replace(s) + "`"
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
