// verify checks that a load test's requests were recorded correctly by each gateway: spend logs,
// the key, team, user and membership spend and the daily tables in Postgres, the Prometheus
// counters, and the S3 request logs. Run with genconfig -features full and one key set per side,
// so every side's records can be told apart in the shared database. A side named bifrost is read
// in Bifrost's schema (see bifrost.go); with three sides each pair is compared.
//
//	verify expect -out expected-python.json results/python-*.jsonl probes-python.jsonl
//	verify sql -set python                   # the query whose one JSON row is db-python.json
//	verify check -dir collected -side python=<metrics host>,<s3 prefix> -side rust=... \
//	    -s3 http://s3stub:9100/bucket -mock http://mockllm:9000 -report report.md
//
// Every side gets the same requests (same loadgen seed, the mock's in= directive fixing input
// tokens), so the sides' totals must also agree with each other, not just with what was sent.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		log.Fatal("usage: verify expect|sql|check ...")
	}
	switch os.Args[1] {
	case "expect":
		expectCmd(os.Args[2:])
	case "sql":
		sqlCmd(os.Args[2:])
	case "check":
		checkCmd(os.Args[2:])
	default:
		log.Fatalf("unknown subcommand %q", os.Args[1])
	}
}

// result is the part of a loadgen record verify reads.
type result struct {
	Model     string `json:"model"`
	Key       int    `json:"key"`
	CtxTokens int64  `json:"ctx"`
	OutTokens int64  `json:"out"`
	OK        bool   `json:"ok"`
}

type tokens struct {
	Requests   int64 `json:"requests"`
	Prompt     int64 `json:"prompt_tokens"`
	Completion int64 `json:"completion_tokens"`
}

func (t *tokens) add(r result) {
	t.Requests++
	t.Prompt += r.CtxTokens
	t.Completion += r.OutTokens
}

// expected is what the gateway was asked to serve successfully.
type expected struct {
	Sent   int64              `json:"sent"`
	Failed int64              `json:"failed"`
	Total  tokens             `json:"total"`
	Groups map[string]*tokens `json:"groups"`
}

func expectCmd(args []string) {
	fs := flag.NewFlagSet("expect", flag.ExitOnError)
	out := fs.String("out", "expected.json", "where to write the totals")
	fs.Parse(args)
	e := expected{Groups: map[string]*tokens{}}
	for _, path := range fs.Args() {
		f, err := os.Open(path)
		if err != nil {
			log.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for sc.Scan() {
			var r result
			if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
				log.Fatalf("%s: %v", path, err)
			}
			e.Sent++
			if !r.OK {
				e.Failed++
				continue
			}
			e.Total.add(r)
			g := e.Groups[r.Model]
			if g == nil {
				g = &tokens{}
				e.Groups[r.Model] = g
			}
			g.add(r)
		}
		f.Close()
	}
	writeJSON(*out, e)
}

func writeJSON(path string, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// check is one line of the report.
type check struct {
	Name, Side, Want, Got string
	OK                    bool
	Note                  string
}

type report struct{ checks []check }

func (r *report) add(c check) { r.checks = append(r.checks, c) }

func (r *report) failed() int {
	n := 0
	for _, c := range r.checks {
		if !c.OK {
			n++
		}
	}
	return n
}

func (r *report) markdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "### Recorded data checks: %d of %d passed\n\n", len(r.checks)-r.failed(), len(r.checks))
	b.WriteString("| check | side | expected | recorded | result |\n|---|---|---|---|---|\n")
	for _, c := range r.checks {
		mark := "ok"
		if !c.OK {
			mark = "**FAIL**"
		}
		if c.Note != "" {
			mark += " (" + c.Note + ")"
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s |\n", c.Name, c.Side, c.Want, c.Got, mark)
	}
	return b.String()
}

// side is one gateway under test.
type side struct {
	name, metricsHost, s3Prefix string
}

type sideFlags []side

func (s *sideFlags) String() string { return fmt.Sprint(*s) }

func (s *sideFlags) Set(v string) error {
	name, rest, ok := strings.Cut(v, "=")
	host, prefix, ok2 := strings.Cut(rest, ",")
	if !ok || !ok2 || name == "" {
		return fmt.Errorf("want name=<metrics host:port>,<s3 prefix>, got %q", v)
	}
	*s = append(*s, side{name: name, metricsHost: host, s3Prefix: strings.Trim(prefix, "/")})
	return nil
}

func checkCmd(args []string) {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	dir := fs.String("dir", ".", "directory with expected-<side>.json and db-<side>.json")
	s3 := fs.String("s3", "", "S3 bucket URL readable without credentials (empty skips S3)")
	mock := fs.String("mock", "", "mock base URL whose /stats counts request headers (empty skips)")
	reportPath := fs.String("report", "report.md", "markdown report")
	var sides sideFlags
	fs.Var(&sides, "side", "name=<metrics host:port>,<s3 prefix> (repeatable)")
	fs.Parse(args)

	var r report
	dbs := map[string]*dbTotals{}
	metrics := map[string]map[string]*family{}
	logs := map[string]*s3Totals{}
	var headers map[string]map[string]int64
	if *mock != "" {
		h, err := mockHeaders(*mock)
		if err != nil {
			r.add(check{Name: "mock header counts read", Side: "mock", Want: "/stats", Got: err.Error()})
		}
		headers = h
	}
	for _, s := range sides {
		var e expected
		if err := readJSON(filepath.Join(*dir, "expected-"+s.name+".json"), &e); err != nil {
			log.Fatal(err)
		}
		var db dbTotals
		if err := readJSON(filepath.Join(*dir, "db-"+s.name+".json"), &db); err != nil {
			log.Fatal(err)
		}
		dbs[s.name] = &db
		checkDB(&r, s.name, e, db)
		if headers != nil {
			checkHeaders(&r, s.name, e, headers)
		}
		if s.metricsHost != "" {
			m, err := scrapeAll(s.metricsHost)
			if err != nil {
				r.add(check{Name: "metrics scraped", Side: s.name, Want: "every replica", Got: err.Error()})
			} else {
				metrics[s.name] = m
				checkMetrics(&r, s.name, e, db, m)
			}
		}
		if *s3 != "" {
			t, err := listLogs(*s3, s.s3Prefix, isBifrost(s.name))
			if err != nil {
				r.add(check{Name: "S3 logs listed", Side: s.name, Want: "a listing", Got: err.Error()})
			} else {
				logs[s.name] = t
				checkLogs(&r, s.name, e, db, t)
			}
		}
	}
	for i := range sides {
		for _, other := range sides[i+1:] {
			a, b := sides[i].name, other.name
			compareSides(&r, a, b, dbs[a], dbs[b], metrics[a], metrics[b], logs[a], logs[b])
		}
	}
	md := r.markdown()
	if err := os.WriteFile(*reportPath, []byte(md), 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Print(md)
	if r.failed() > 0 {
		os.Exit(1)
	}
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
