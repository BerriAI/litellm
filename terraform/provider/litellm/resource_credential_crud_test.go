package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// newTestResourceData creates a *schema.ResourceData with the credential schema,
// sets the ID and populates the required fields.
func newTestResourceData(t *testing.T, id string) *schema.ResourceData {
	t.Helper()
	d := schema.TestResourceDataRaw(t, resourceLiteLLMCredential().Schema, map[string]interface{}{
		"credential_name":   id,
		"model_id":          "",
		"credential_info":   map[string]interface{}{},
		"credential_values": map[string]interface{}{"key": "val"},
	})
	d.SetId(id)
	return d
}

func TestRetryCredentialRead_SuccessOnFirstAttempt(t *testing.T) {
	resp := CredentialResponse{
		CredentialName: "test-cred",
		CredentialInfo: map[string]interface{}{"provider": "aws"},
	}
	body, _ := json.Marshal(resp)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(body)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTestResourceData(t, "test-cred")

	err := retryCredentialRead(d, client, 3)
	if err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "test-cred" {
		t.Fatalf("expected ID 'test-cred', got %q", d.Id())
	}
}

func TestRetryCredentialRead_SuccessAfterRetries(t *testing.T) {
	resp := CredentialResponse{
		CredentialName: "test-cred",
		CredentialInfo: map[string]interface{}{"provider": "aws"},
	}
	body, _ := json.Marshal(resp)

	var callCount int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		w.Header().Set("Content-Type", "application/json")
		if n <= 2 {
			// First two calls return 404, triggering retry
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(body)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTestResourceData(t, "test-cred")

	err := retryCredentialRead(d, client, 3)
	if err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "test-cred" {
		t.Fatalf("expected ID 'test-cred', got %q", d.Id())
	}
	if atomic.LoadInt32(&callCount) != 3 {
		t.Fatalf("expected 3 HTTP calls, got %d", callCount)
	}
}

func TestRetryCredentialRead_ExhaustsRetries(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTestResourceData(t, "test-cred")

	err := retryCredentialRead(d, client, 2)
	if err == nil {
		t.Fatal("expected error after exhausting retries, got nil")
	}
	if err.Error() != "credential_not_found" {
		t.Fatalf("expected 'credential_not_found' error, got: %v", err)
	}
	// ID should still be restored (not wiped)
	if d.Id() != "test-cred" {
		t.Fatalf("expected ID to be restored to 'test-cred', got %q", d.Id())
	}
}

func TestRetryCredentialRead_NonRetryableError(t *testing.T) {
	var callCount int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "internal server error"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTestResourceData(t, "test-cred")

	err := retryCredentialRead(d, client, 3)
	if err == nil {
		t.Fatal("expected error for 500 response, got nil")
	}
	// Should fail on first attempt without retrying
	if atomic.LoadInt32(&callCount) != 1 {
		t.Fatalf("expected 1 HTTP call (no retries for non-retryable error), got %d", callCount)
	}
}

func TestRetryCredentialRead_IDRestoredBetweenRetries(t *testing.T) {
	// Verify the ID is restored after each failed attempt where the read clears it.
	// resourceLiteLLMCredentialRead sets ID to "" on 404, and retryCredentialRead
	// should restore it before the next attempt.
	resp := CredentialResponse{
		CredentialName: "my-cred",
		CredentialInfo: map[string]interface{}{},
	}
	body, _ := json.Marshal(resp)

	var callCount int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(body)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTestResourceData(t, "my-cred")

	err := retryCredentialRead(d, client, 2)
	if err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if d.Id() != "my-cred" {
		t.Fatalf("expected ID 'my-cred', got %q", d.Id())
	}
}

func TestRetryCredentialRead_MaxRetriesOne(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTestResourceData(t, "test-cred")

	err := retryCredentialRead(d, client, 1)
	if err == nil {
		t.Fatal("expected error with maxRetries=1 and always-404, got nil")
	}
	if err.Error() != "credential_not_found" {
		t.Fatalf("expected 'credential_not_found', got: %v", err)
	}
}

func TestRetryCredentialRead_ConnectionError(t *testing.T) {
	// Point to a server that's already closed to simulate connection failure
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := newTestResourceData(t, "test-cred")

	err := retryCredentialRead(d, client, 1)
	if err == nil {
		t.Fatal("expected error for connection failure, got nil")
	}
	// Connection error should not be retried (not a "credential_not_found")
	fmt.Printf("connection error (expected): %v\n", err)
}
