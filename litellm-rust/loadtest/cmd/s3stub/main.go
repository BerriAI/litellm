// s3stub is an in-memory, path-style S3 endpoint for load tests: PUT stores an object, GET reads
// it, and GET /<bucket>?list-type=2 lists keys (ListObjectsV2, 1,000 per page). It checks no
// credentials, so gateways can sign their uploads as they would for AWS and verify can read them
// back without any. Objects live until the process exits.
//
// -max-string N shortens every JSON string longer than N bytes in a stored object to its first N
// bytes, so request logs keep their fields and numbers but not whole prompts, and a long run's logs
// fit in memory. Objects that are not JSON are stored as sent.
//
// Objects are kept deflate-compressed and listed from a sorted index that is rebuilt only after
// new keys arrive, so a 15-minute run at 1,000 requests/s per gateway (millions of logs) fits in a
// few GiB and verify's listing stays linear.
//
//	s3stub -addr :9100 -max-string 256
package main

import (
	"bytes"
	"compress/flate"
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
	objects   map[string]object // "<bucket>/<key>"
	sorted    []string          // every key of objects, sorted; nil after a new key arrives
	maxString int
}

// object is a stored body, deflate-compressed, and its size as sent.
type object struct {
	data []byte
	size int
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
	s := &store{objects: map[string]object{}, maxString: *maxString}
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
		obj := object{data: compress(body), size: len(body)}
		s.mu.Lock()
		if _, ok := s.objects[bucket+"/"+key]; !ok {
			s.sorted = nil
		}
		s.objects[bucket+"/"+key] = obj
		s.mu.Unlock()
		w.Header().Set("ETag", `"stub"`)
	case r.Method == http.MethodGet && key == "":
		s.list(w, r, bucket)
	case r.Method == http.MethodHead && key == "":
		// HeadBucket: every bucket exists. Bifrost checks its log bucket at startup.
	case (r.Method == http.MethodGet || r.Method == http.MethodHead) && key != "":
		s.mu.RLock()
		obj, ok := s.objects[bucket+"/"+key]
		s.mu.RUnlock()
		if !ok {
			http.Error(w, "NoSuchKey", http.StatusNotFound)
			return
		}
		w.Header().Set("content-length", strconv.Itoa(obj.size))
		if r.Method == http.MethodGet {
			io.Copy(w, flate.NewReader(bytes.NewReader(obj.data)))
		}
	default:
		http.Error(w, "unsupported", http.StatusMethodNotAllowed)
	}
}

func (s *store) list(w http.ResponseWriter, r *http.Request, bucket string) {
	prefix := r.URL.Query().Get("prefix")
	after := r.URL.Query().Get("continuation-token")
	res := listResult{Name: bucket, Prefix: prefix}
	s.mu.Lock()
	if s.sorted == nil {
		s.sorted = make([]string, 0, len(s.objects))
		for full := range s.objects {
			s.sorted = append(s.sorted, full)
		}
		sort.Strings(s.sorted)
	}
	from := bucket + "/" + prefix
	i := sort.SearchStrings(s.sorted, from)
	if after != "" {
		// Past every key up to and including the token.
		i = max(i, sort.Search(len(s.sorted), func(j int) bool { return s.sorted[j] > bucket+"/"+after }))
	}
	for ; i < len(s.sorted) && strings.HasPrefix(s.sorted[i], from); i++ {
		if len(res.Contents) == page {
			res.IsTruncated = true
			res.NextContinuationToken = res.Contents[page-1].Key
			break
		}
		key := strings.TrimPrefix(s.sorted[i], bucket+"/")
		res.Contents = append(res.Contents, entry{Key: key, Size: s.objects[s.sorted[i]].size})
	}
	s.mu.Unlock()
	res.KeyCount = len(res.Contents)
	w.Header().Set("content-type", "application/xml")
	io.WriteString(w, xml.Header)
	xml.NewEncoder(w).Encode(res)
}

func compress(body []byte) []byte {
	var b bytes.Buffer
	zw, _ := flate.NewWriter(&b, flate.BestSpeed)
	zw.Write(body)
	zw.Close()
	return b.Bytes()
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
