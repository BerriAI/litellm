package main

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// maxLogs bounds how many request logs one side's check downloads.
const maxLogs = 20_000

// s3Totals sums one side's request logs: one StandardLoggingPayload object per request.
type s3Totals struct {
	Objects    int64
	Read       int64
	Success    int64
	Prompt     int64
	Completion int64
	Cost       float64
	WithBodies int64
	WithTeam   int64
	Fields     map[string]bool
	Metadata   map[string]bool
}

type listing struct {
	Contents []struct {
		Key string `xml:"Key"`
	} `xml:"Contents"`
	IsTruncated           bool   `xml:"IsTruncated"`
	NextContinuationToken string `xml:"NextContinuationToken"`
}

// listLogs lists bucket objects under prefix (ListObjectsV2, no credentials: the bucket allows
// anonymous reads) and reads up to maxLogs of them.
func listLogs(bucket, prefix string) (*s3Totals, error) {
	client := &http.Client{Timeout: 60 * time.Second}
	bucket = strings.TrimRight(bucket, "/")
	var keys []string
	token := ""
	for {
		q := url.Values{"list-type": {"2"}, "prefix": {prefix + "/"}}
		if token != "" {
			q.Set("continuation-token", token)
		}
		resp, err := client.Get(bucket + "?" + q.Encode())
		if err != nil {
			return nil, err
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("list answered %d: %.200s", resp.StatusCode, body)
		}
		var l listing
		if err := xml.Unmarshal(body, &l); err != nil {
			return nil, err
		}
		for _, c := range l.Contents {
			keys = append(keys, c.Key)
		}
		if !l.IsTruncated || l.NextContinuationToken == "" {
			break
		}
		token = l.NextContinuationToken
	}
	sort.Strings(keys)
	t := &s3Totals{Objects: int64(len(keys)), Fields: map[string]bool{}, Metadata: map[string]bool{}}
	for i, key := range keys {
		if i >= maxLogs {
			break
		}
		resp, err := client.Get(bucket + "/" + (&url.URL{Path: key}).EscapedPath())
		if err != nil {
			return nil, err
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("%s answered %d", key, resp.StatusCode)
		}
		// One payload per object, or one per line with batched uploads.
		for _, line := range strings.Split(strings.TrimSpace(string(body)), "\n") {
			if err := t.add([]byte(line)); err != nil {
				return nil, fmt.Errorf("%s: %v", key, err)
			}
		}
	}
	return t, nil
}

func (t *s3Totals) add(raw []byte) error {
	var p map[string]any
	if err := json.Unmarshal(raw, &p); err != nil {
		return err
	}
	t.Read++
	for k := range p {
		t.Fields[k] = true
	}
	number := func(v any) float64 {
		f, _ := v.(float64)
		return f
	}
	if p["status"] == "success" {
		t.Success++
	}
	t.Prompt += int64(number(p["prompt_tokens"]))
	t.Completion += int64(number(p["completion_tokens"]))
	t.Cost += number(p["response_cost"])
	if p["messages"] != nil && p["response"] != nil {
		t.WithBodies++
	}
	if m, ok := p["metadata"].(map[string]any); ok {
		for k := range m {
			t.Metadata[k] = true
		}
		if team, _ := m["user_api_key_team_id"].(string); team != "" {
			t.WithTeam++
		}
	}
	return nil
}

func checkLogs(r *report, side string, e expected, db dbTotals, t *s3Totals) {
	ok := e.Total.Requests
	exact(r, "S3 request logs", side, ok, t.Objects)
	if t.Read < t.Objects {
		r.add(check{Name: "S3 logs read", Side: side, Want: fmt.Sprint(t.Objects), Got: fmt.Sprint(t.Read),
			OK: true, Note: fmt.Sprintf("totals below cover the first %d", t.Read)})
		return
	}
	exact(r, "S3 logs with status success", side, ok, t.Success)
	exact(r, "S3 logs with messages and response", side, ok, t.WithBodies)
	exact(r, "S3 logs naming the key's team", side, ok, t.WithTeam)
	exact(r, "S3 log prompt tokens", side, e.Total.Prompt, t.Prompt)
	exact(r, "S3 log completion tokens", side, e.Total.Completion, t.Completion)
	dollars(r, "S3 log response_cost = spend logs", side, db.Spend, t.Cost)
}
