// s3stub is an in-memory, path-style S3 endpoint for load tests: PUT stores an object, GET reads
// it, and GET /<bucket>?list-type=2 lists keys (ListObjectsV2, 1,000 per page). It checks no
// credentials, so gateways can sign their uploads as they would for AWS and verify can read them
// back without any. Objects live until the process exits.
//
// -max-string N shortens every JSON string longer than N bytes in a stored object to its first N
// bytes, so request logs keep their fields and numbers but not whole prompts, and a long run's logs
// fit in memory. Objects that are not JSON are stored as sent.
//
//	s3stub -addr :9100 -max-string 256
package main

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"flag"
	"io"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
)

type store struct {
	mu        sync.RWMutex
	objects   map[string][]byte // "<bucket>/<key>"
	maxString int
}

type listResult struct {
	XMLName               xml.Name `xml:"ListBucketResult"`
	Name                  string   `xml:"Name"`
	Prefix                string   `xml:"Prefix"`
	KeyCount              int      `xml:"KeyCount"`
	IsTruncated           bool     `xml:"IsTruncated"`
	NextContinuationToken string   `xml:"NextContinuationToken,omitempty"`
	Contents              []entry  `xml:"Contents"`
}

type entry struct {
	Key  string `xml:"Key"`
	Size int    `xml:"Size"`
}

const page = 1000

func main() {
	addr := flag.String("addr", ":9100", "listen address")
	maxString := flag.Int("max-string", 0, "shorten JSON strings longer than this many bytes (0 keeps objects as sent)")
	flag.Parse()
	s := &store{objects: map[string][]byte{}, maxString: *maxString}
	http.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok") })
	http.HandleFunc("/", s.serve)
	log.Printf("s3stub listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}

func (s *store) serve(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/")
	bucket, key, _ := strings.Cut(path, "/")
	if bucket == "" {
		http.Error(w, "no bucket", http.StatusBadRequest)
		return
	}
	switch {
	case r.Method == http.MethodPut && key != "":
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if s.maxString > 0 {
			body = shorten(body, s.maxString)
		}
		s.mu.Lock()
		s.objects[bucket+"/"+key] = body
		s.mu.Unlock()
		w.Header().Set("ETag", `"stub"`)
	case r.Method == http.MethodGet && key == "":
		s.list(w, r, bucket)
	case r.Method == http.MethodHead && key == "":
		// HeadBucket: every bucket exists. Bifrost checks its log bucket at startup.
	case (r.Method == http.MethodGet || r.Method == http.MethodHead) && key != "":
		s.mu.RLock()
		body, ok := s.objects[bucket+"/"+key]
		s.mu.RUnlock()
		if !ok {
			http.Error(w, "NoSuchKey", http.StatusNotFound)
			return
		}
		w.Header().Set("content-length", strconv.Itoa(len(body)))
		if r.Method == http.MethodGet {
			w.Write(body)
		}
	default:
		http.Error(w, "unsupported", http.StatusMethodNotAllowed)
	}
}

func (s *store) list(w http.ResponseWriter, r *http.Request, bucket string) {
	prefix := r.URL.Query().Get("prefix")
	after := r.URL.Query().Get("continuation-token")
	s.mu.RLock()
	var keys []string
	sizes := map[string]int{}
	for full, body := range s.objects {
		b, key, _ := strings.Cut(full, "/")
		if b == bucket && strings.HasPrefix(key, prefix) && key > after {
			keys = append(keys, key)
			sizes[key] = len(body)
		}
	}
	s.mu.RUnlock()
	sort.Strings(keys)
	res := listResult{Name: bucket, Prefix: prefix}
	if len(keys) > page {
		res.IsTruncated = true
		res.NextContinuationToken = keys[page-1]
		keys = keys[:page]
	}
	for _, k := range keys {
		res.Contents = append(res.Contents, entry{Key: k, Size: sizes[k]})
	}
	res.KeyCount = len(res.Contents)
	w.Header().Set("content-type", "application/xml")
	io.WriteString(w, xml.Header)
	xml.NewEncoder(w).Encode(res)
}

// shorten re-encodes a JSON object with every string cut to max bytes; numbers keep their text.
func shorten(body []byte, max int) []byte {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil || dec.More() {
		return body
	}
	out, err := json.Marshal(cut(v, max))
	if err != nil {
		return body
	}
	return out
}

func cut(v any, max int) any {
	switch v := v.(type) {
	case string:
		if len(v) > max {
			return v[:max]
		}
	case []any:
		for i := range v {
			v[i] = cut(v[i], max)
		}
	case map[string]any:
		for k := range v {
			v[k] = cut(v[k], max)
		}
	}
	return v
}
