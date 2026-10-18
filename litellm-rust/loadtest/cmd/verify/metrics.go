package main

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

// family is one metric name summed over every series and replica.
type family struct {
	sum    float64
	labels map[string]bool
}

// scrapeAll reads /metrics from every address host resolves to (a headless Service names each
// replica) and sums each sample name across them.
func scrapeAll(hostPort string) (map[string]*family, error) {
	host, port, err := net.SplitHostPort(hostPort)
	if err != nil {
		return nil, err
	}
	addrs, err := net.LookupHost(host)
	if err != nil {
		return nil, err
	}
	sort.Strings(addrs)
	client := &http.Client{Timeout: 30 * time.Second}
	out := map[string]*family{}
	for _, addr := range addrs {
		resp, err := client.Get("http://" + net.JoinHostPort(addr, port) + "/metrics")
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, fmt.Errorf("%s: /metrics answered %d", addr, resp.StatusCode)
		}
		err = parseMetrics(resp.Body, out)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// parseMetrics adds each sample of a Prometheus text exposition to out, by sample name.
func parseMetrics(r io.Reader, out map[string]*family) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, labels, rest := line, "", ""
		if i := strings.IndexByte(line, '{'); i >= 0 {
			j := strings.LastIndexByte(line, '}')
			if j < i {
				return fmt.Errorf("malformed sample %q", line)
			}
			name, labels, rest = line[:i], line[i+1:j], strings.TrimSpace(line[j+1:])
		} else if i := strings.IndexByte(line, ' '); i >= 0 {
			name, rest = line[:i], strings.TrimSpace(line[i+1:])
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			return fmt.Errorf("sample without a value %q", line)
		}
		value, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			return fmt.Errorf("sample value in %q: %v", line, err)
		}
		f := out[name]
		if f == nil {
			f = &family{labels: map[string]bool{}}
			out[name] = f
		}
		f.sum += value
		for _, pair := range splitLabels(labels) {
			if k, _, ok := strings.Cut(pair, "="); ok {
				f.labels[strings.TrimSpace(k)] = true
			}
		}
	}
	return sc.Err()
}

// splitLabels splits `a="x",b="y,z"` on the commas outside quotes.
func splitLabels(s string) []string {
	var parts []string
	quoted, escaped, start := false, false, 0
	for i, c := range s {
		switch {
		case escaped:
			escaped = false
		case c == '\\':
			escaped = true
		case c == '"':
			quoted = !quoted
		case c == ',' && !quoted:
			parts = append(parts, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		parts = append(parts, s[start:])
	}
	return parts
}

func sample(m map[string]*family, name string) (float64, bool) {
	f, ok := m[name]
	if !ok {
		return 0, false
	}
	return f.sum, true
}

func checkMetrics(r *report, side string, e expected, db dbTotals, m map[string]*family) {
	if isBifrost(side) {
		checkBifrostMetrics(r, side, e, db, m)
		return
	}
	counts := []struct {
		name string
		want int64
	}{
		{"litellm_proxy_total_requests_metric_total", e.Sent},
		{"litellm_request_total_latency_metric_count", e.Total.Requests},
		{"litellm_llm_api_latency_metric_count", e.Total.Requests},
		{"litellm_deployment_success_responses_total", e.Total.Requests},
		{"litellm_input_tokens_metric_total", e.Total.Prompt},
		{"litellm_output_tokens_metric_total", e.Total.Completion},
		{"litellm_total_tokens_metric_total", e.Total.Prompt + e.Total.Completion},
	}
	for _, c := range counts {
		got, ok := sample(m, c.name)
		r.add(check{Name: "metric " + c.name, Side: side, Want: fmt.Sprint(c.want),
			Got: present(ok, fmt.Sprintf("%.0f", got)), OK: ok && got == float64(c.want)})
	}
	got, ok := sample(m, "litellm_spend_metric_total")
	r.add(check{Name: "metric litellm_spend_metric_total = spend logs", Side: side,
		Want: fmt.Sprintf("$%.6f", db.Spend), Got: present(ok, fmt.Sprintf("$%.6f", got)), OK: ok && near(db.Spend, got)})
}

func present(ok bool, value string) string {
	if !ok {
		return "missing"
	}
	return value
}

// litellmFamilies lists the sample names a gateway serves for its requests, so the sides can be
// compared without process and runtime metrics.
func litellmFamilies(m map[string]*family) map[string]*family {
	out := map[string]*family{}
	for name, f := range m {
		// prometheus_client's _created series only timestamp the counters beside them.
		if strings.HasPrefix(name, "litellm_") && !strings.HasSuffix(name, "_created") {
			out[name] = f
		}
	}
	return out
}
