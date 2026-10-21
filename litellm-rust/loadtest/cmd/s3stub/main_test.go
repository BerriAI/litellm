package main

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestShortenKeepsFieldsAndNumbers(t *testing.T) {
	in := []byte(`{"messages":[{"content":"abcdefgh"}],"response_cost":0.10262000000000001,"n":3,"ok":true}`)
	got := string(shorten(in, 3))
	want := `{"messages":[{"content":"abc"}],"n":3,"ok":true,"response_cost":0.10262000000000001}`
	if got != want {
		t.Fatalf("got %s", got)
	}
	if got := string(shorten([]byte("not json"), 3)); got != "not json" {
		t.Fatalf("non-JSON changed: %s", got)
	}
}

func TestPutGetAndListPages(t *testing.T) {
	s := &store{objects: map[string]object{}}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	defer srv.Close()
	put := func(key, body string) {
		req, _ := http.NewRequest(http.MethodPut, srv.URL+"/logs/"+key, strings.NewReader(body))
		resp, err := http.DefaultClient.Do(req)
		if err != nil || resp.StatusCode != http.StatusOK {
			t.Fatalf("put %s: %v %v", key, err, resp)
		}
		resp.Body.Close()
	}
	const n = 2*page + 7
	for i := 0; i < n; i++ {
		put(fmt.Sprintf("rust/%05d.json", i), fmt.Sprintf(`{"i":%d}`, i))
	}
	put("python/00000.json", `{"i":-1}`)

	resp, err := http.Get(srv.URL + "/logs/rust/00042.json")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(body) != `{"i":42}` || resp.Header.Get("content-length") != "8" {
		t.Fatalf("get: %q %s", body, resp.Header.Get("content-length"))
	}

	var keys []string
	token := ""
	for {
		q := url.Values{"list-type": {"2"}, "prefix": {"rust/"}}
		if token != "" {
			q.Set("continuation-token", token)
		}
		resp, err := http.Get(srv.URL + "/logs?" + q.Encode())
		if err != nil {
			t.Fatal(err)
		}
		var l listResult
		if err := xml.NewDecoder(resp.Body).Decode(&l); err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		for _, c := range l.Contents {
			keys = append(keys, c.Key)
		}
		if !l.IsTruncated {
			break
		}
		token = l.NextContinuationToken
	}
	if len(keys) != n || keys[0] != "rust/00000.json" || keys[n-1] != fmt.Sprintf("rust/%05d.json", n-1) {
		t.Fatalf("listed %d keys, first %s last %s", len(keys), keys[0], keys[len(keys)-1])
	}
	for i := 1; i < len(keys); i++ {
		if keys[i] <= keys[i-1] {
			t.Fatalf("keys out of order at %d: %s after %s", i, keys[i], keys[i-1])
		}
	}
}
