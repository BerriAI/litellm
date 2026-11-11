package litellm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func newFallbackTestResourceData(t *testing.T, model string, fallbackModels []interface{}, fallbackType string) *schema.ResourceData {
	t.Helper()
	d := schema.TestResourceDataRaw(t, resourceLiteLLMFallback().Schema, map[string]interface{}{
		"model":           model,
		"fallback_models": fallbackModels,
		"fallback_type":   fallbackType,
	})
	return d
}

func fallbackGetHandler(t *testing.T, wantPath string, resp FallbackGetResponse) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != wantPath {
			t.Errorf("expected path %s, got %s", wantPath, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func TestResourceLiteLLMFallbackCreate(t *testing.T) {
	var createPayload map[string]interface{}
	mux := http.NewServeMux()
	mux.HandleFunc("/fallback", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
			t.Fatalf("failed to decode create payload: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_models":["claude-3","gpt-3.5-turbo"],"fallback_type":"general","message":"ok"}`))
	})
	mux.Handle("/fallback/gpt-4", fallbackGetHandler(t, "/fallback/gpt-4", FallbackGetResponse{
		Model:          "gpt-4",
		FallbackModels: []string{"claude-3", "gpt-3.5-turbo"},
		FallbackType:   "general",
	}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3", "gpt-3.5-turbo"}, "general")

	if diags := resourceLiteLLMFallbackCreate(context.Background(), d, client); diags.HasError() {
		t.Fatalf("expected no error, got: %v", diags)
	}
	if d.Id() != "gpt-4" {
		t.Fatalf("expected ID 'gpt-4', got %q", d.Id())
	}
	want := map[string]interface{}{
		"model":           "gpt-4",
		"fallback_models": []interface{}{"claude-3", "gpt-3.5-turbo"},
		"fallback_type":   "general",
	}
	if !reflect.DeepEqual(createPayload, want) {
		t.Fatalf("unexpected create payload: %+v, want %+v", createPayload, want)
	}
}

func TestResourceLiteLLMFallbackRead_MapsFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/fallback/gpt-4" {
			t.Errorf("expected path /fallback/gpt-4, got %s", r.URL.Path)
		}
		if got := r.URL.Query().Get("fallback_type"); got != "context_window" {
			t.Errorf("expected fallback_type query 'context_window', got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_models":["claude-3"],"fallback_type":"context_window"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"stale-model"}, "context_window")
	d.SetId("gpt-4")

	if err := resourceLiteLLMFallbackRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	got := d.Get("fallback_models").([]interface{})
	if !reflect.DeepEqual(got, []interface{}{"claude-3"}) {
		t.Fatalf("expected fallback_models [claude-3], got %+v", got)
	}
	if d.Get("fallback_type").(string) != "context_window" {
		t.Fatalf("expected fallback_type 'context_window', got %q", d.Get("fallback_type"))
	}
	if d.Get("model").(string) != "gpt-4" {
		t.Fatalf("expected model 'gpt-4', got %q", d.Get("model"))
	}
}

func TestResourceLiteLLMFallbackRead_404ClearsID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")
	d.SetId("gpt-4")

	if err := resourceLiteLLMFallbackRead(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared on 404, got %q", d.Id())
	}
}

func TestResourceLiteLLMFallbackUpdate_SendsChangedModels(t *testing.T) {
	var updatePayload map[string]interface{}
	mux := http.NewServeMux()
	mux.HandleFunc("/fallback", func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&updatePayload)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_models":["new-model"],"fallback_type":"general","message":"ok"}`))
	})
	mux.Handle("/fallback/gpt-4", fallbackGetHandler(t, "/fallback/gpt-4", FallbackGetResponse{
		Model:          "gpt-4",
		FallbackModels: []string{"new-model"},
		FallbackType:   "general",
	}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"new-model"}, "general")
	d.SetId("gpt-4")

	if diags := resourceLiteLLMFallbackUpdate(context.Background(), d, client); diags.HasError() {
		t.Fatalf("expected no error, got: %v", diags)
	}
	if !reflect.DeepEqual(updatePayload["fallback_models"], []interface{}{"new-model"}) {
		t.Fatalf("expected updated fallback_models [new-model], got %+v", updatePayload["fallback_models"])
	}
}

func TestResourceLiteLLMFallbackDelete(t *testing.T) {
	var gotMethod, gotPath, gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotType = r.URL.Query().Get("fallback_type")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","fallback_type":"general","message":"deleted"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")
	d.SetId("gpt-4")

	if err := resourceLiteLLMFallbackDelete(d, client); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if gotMethod != http.MethodDelete || gotPath != "/fallback/gpt-4" || gotType != "general" {
		t.Fatalf("expected DELETE /fallback/gpt-4?fallback_type=general, got %s %s?fallback_type=%s",
			gotMethod, gotPath, gotType)
	}
	if d.Id() != "" {
		t.Fatalf("expected ID cleared after delete, got %q", d.Id())
	}
}

type fallbackRead struct {
	status int
	models []string
}

type fallbackProxy struct {
	*httptest.Server
	posts atomic.Int32
	reads atomic.Int32
}

func fallbackServer(t *testing.T, reads []fallbackRead) *fallbackProxy {
	t.Helper()
	proxy := &fallbackProxy{}
	mux := http.NewServeMux()
	mux.HandleFunc("/fallback", func(w http.ResponseWriter, r *http.Request) {
		proxy.posts.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"message":"ok"}`))
	})
	mux.HandleFunc("/fallback/gpt-4", func(w http.ResponseWriter, r *http.Request) {
		index := int(proxy.reads.Add(1)) - 1
		if index >= len(reads) {
			t.Errorf("unexpected read %d after the proxy served the fallback", index+1)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		read := reads[index]
		if read.status != http.StatusOK {
			w.WriteHeader(read.status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(FallbackGetResponse{Model: "gpt-4", FallbackModels: read.models, FallbackType: "general"})
	})
	proxy.Server = httptest.NewServer(mux)
	t.Cleanup(proxy.Close)
	return proxy
}

type recordedWaits struct {
	delays []time.Duration
	failAt int
}

func (w *recordedWaits) wait(ctx context.Context, delay time.Duration) error {
	w.delays = append(w.delays, delay)
	if w.failAt > 0 && len(w.delays) >= w.failAt {
		return context.DeadlineExceeded
	}
	return nil
}

func TestFallbackWriteWaitsForTheProxyToServeIt(t *testing.T) {
	cases := []struct {
		name   string
		action string
		reads  []fallbackRead
	}{
		{
			name:   "create answered by a replica without the new rule",
			action: "creating",
			reads: []fallbackRead{
				{status: http.StatusNotFound},
				{status: http.StatusNotFound},
				{status: http.StatusOK, models: []string{"claude-3", "gpt-3.5-turbo"}},
			},
		},
		{
			name:   "update answered by a replica with the previous list",
			action: "updating",
			reads: []fallbackRead{
				{status: http.StatusOK, models: []string{"gpt-3.5-turbo", "claude-3"}},
				{status: http.StatusOK, models: []string{"claude-3"}},
				{status: http.StatusOK, models: []string{"claude-3", "gpt-3.5-turbo"}},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			proxy := fallbackServer(t, tc.reads)
			waits := &recordedWaits{}
			d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3", "gpt-3.5-turbo"}, "general")

			err := upsertLiteLLMFallback(context.Background(), d, NewClient(proxy.URL, "test-key", true), tc.action, waits.wait)

			if err != nil {
				t.Fatalf("expected the write to succeed once the proxy serves it, got: %v", err)
			}
			if d.Id() != "gpt-4" {
				t.Fatalf("expected ID gpt-4, got %q", d.Id())
			}
			if got := d.Get("fallback_models"); !reflect.DeepEqual(got, []interface{}{"claude-3", "gpt-3.5-turbo"}) {
				t.Fatalf("expected state to hold the written list, got %v", got)
			}
			if proxy.reads.Load() != int32(len(tc.reads)) || len(waits.delays) != len(tc.reads)-1 {
				t.Fatalf("expected %d reads with a wait between each, got %d reads and waits %v", len(tc.reads), proxy.reads.Load(), waits.delays)
			}
		})
	}
}

func TestFallbackReadBackBacksOffUpToTheMaxDelay(t *testing.T) {
	reads := make([]fallbackRead, 7)
	for i := range reads {
		reads[i] = fallbackRead{status: http.StatusNotFound}
	}
	proxy := fallbackServer(t, append(reads, fallbackRead{status: http.StatusOK, models: []string{"claude-3"}}))
	waits := &recordedWaits{}
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")

	if err := upsertLiteLLMFallback(context.Background(), d, NewClient(proxy.URL, "test-key", true), "creating", waits.wait); err != nil {
		t.Fatalf("expected success, got: %v", err)
	}

	want := []time.Duration{time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 10 * time.Second, 10 * time.Second, 10 * time.Second}
	if !reflect.DeepEqual(waits.delays, want) {
		t.Fatalf("expected waits %v, got %v", want, waits.delays)
	}
}

func TestFallbackWriteThatNeverConvergesKeepsTheIDAndNamesTheModel(t *testing.T) {
	proxy := fallbackServer(t, []fallbackRead{
		{status: http.StatusNotFound},
		{status: http.StatusOK, models: []string{"stale"}},
		{status: http.StatusNotFound},
	})
	waits := &recordedWaits{failAt: 3}
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")

	err := upsertLiteLLMFallback(context.Background(), d, NewClient(proxy.URL, "test-key", true), "creating", waits.wait)

	if !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "gpt-4") {
		t.Fatalf("expected a deadline error naming the model, got: %v", err)
	}
	if d.Id() != "gpt-4" {
		t.Fatalf("expected the ID to stay so the saved fallback is not orphaned, got %q", d.Id())
	}
}

func TestFallbackReadBackDoesNotRetryAnErrorResponse(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusBadRequest, http.StatusInternalServerError} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			proxy := fallbackServer(t, []fallbackRead{{status: status}})
			waits := &recordedWaits{}
			d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")

			err := upsertLiteLLMFallback(context.Background(), d, NewClient(proxy.URL, "test-key", true), "creating", waits.wait)

			if err == nil || !strings.Contains(err.Error(), http.StatusText(status)) {
				t.Fatalf("expected the %d to surface as an error, got: %v", status, err)
			}
			if proxy.reads.Load() != 1 || len(waits.delays) != 0 {
				t.Fatalf("expected one read and no retry, got %d reads and waits %v", proxy.reads.Load(), waits.delays)
			}
			if d.Id() != "gpt-4" {
				t.Fatalf("expected the ID to stay, got %q", d.Id())
			}
		})
	}
}

func TestFallbackWriteSendsNothingOnceApplyIsCancelled(t *testing.T) {
	proxy := fallbackServer(t, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")

	diags := resourceLiteLLMFallbackCreate(ctx, d, NewClient(proxy.URL, "test-key", true))

	if !diags.HasError() || !strings.Contains(diags[0].Summary, context.Canceled.Error()) {
		t.Fatalf("expected a cancellation error, got: %v", diags)
	}
	if proxy.posts.Load() != 0 || proxy.reads.Load() != 0 {
		t.Fatalf("expected no request after cancellation, got %d posts and %d reads", proxy.posts.Load(), proxy.reads.Load())
	}
}

func TestFallbackReadBackStopsWaitingWhenApplyIsCancelled(t *testing.T) {
	proxy := fallbackServer(t, []fallbackRead{{status: http.StatusNotFound}})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cancelThenWait := func(ctx context.Context, delay time.Duration) error {
		cancel()
		return waitOrCancel(ctx, time.Hour)
	}
	d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")

	err := upsertLiteLLMFallback(ctx, d, NewClient(proxy.URL, "test-key", true), "creating", cancelThenWait)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected a cancellation error, got: %v", err)
	}
	if proxy.reads.Load() != 1 || d.Id() != "gpt-4" {
		t.Fatalf("expected one read and the ID kept, got %d reads and ID %q", proxy.reads.Load(), d.Id())
	}
}

func TestFallbackWriteAbortsAHungRequestWhenApplyIsCancelled(t *testing.T) {
	for _, hungPath := range []string{"/fallback", "/fallback/gpt-4"} {
		t.Run(hungPath, func(t *testing.T) {
			arrived := make(chan struct{})
			mux := http.NewServeMux()
			mux.HandleFunc("/fallback", func(w http.ResponseWriter, r *http.Request) {
				if hungPath == "/fallback" {
					close(arrived)
					<-r.Context().Done()
					return
				}
				w.Write([]byte(`{"message":"ok"}`))
			})
			mux.HandleFunc("/fallback/gpt-4", func(w http.ResponseWriter, r *http.Request) {
				close(arrived)
				<-r.Context().Done()
			})
			srv := httptest.NewServer(mux)
			defer srv.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go func() {
				<-arrived
				cancel()
			}()
			d := newFallbackTestResourceData(t, "gpt-4", []interface{}{"claude-3"}, "general")

			err := upsertLiteLLMFallback(ctx, d, NewClient(srv.URL, "test-key", true), "creating", waitOrCancel)

			if !errors.Is(err, context.Canceled) {
				t.Fatalf("expected the hung request to end with the cancellation, got: %v", err)
			}
		})
	}
}
