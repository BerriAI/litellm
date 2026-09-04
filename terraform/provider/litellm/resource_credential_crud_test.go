package litellm

import (
	"encoding/json"
	"fmt"
	"io"
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

// conflictServer builds the shared conflict-then-recover mock used by the
// adoption tests below. patchStatus/patchBody control the PATCH response, so
// callers can exercise both the success and failure paths.
func conflictServer(t *testing.T, patchStatus int, patchBody string) (*httptest.Server, *int32, *int32, *[]byte) {
	t.Helper()
	var createCalls, patchCalls int32
	var capturedPatchBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/credentials":
			atomic.AddInt32(&createCalls, 1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error":{"message":"Unique constraint failed on the fields: (` + "`credential_name`" + `)","type":"internal_server_error","code":"500"}}`))
		case r.Method == http.MethodPatch:
			atomic.AddInt32(&patchCalls, 1)
			if r.URL.Path != "/credentials/conflict-test" {
				t.Errorf("PATCH went to %q, want /credentials/conflict-test", r.URL.Path)
			}
			body, _ := io.ReadAll(r.Body)
			capturedPatchBody = body
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(patchStatus)
			w.Write([]byte(patchBody))
		case r.Method == http.MethodGet:
			resp := CredentialResponse{CredentialName: "conflict-test", CredentialInfo: map[string]interface{}{}}
			body, _ := json.Marshal(resp)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(body)
		default:
			http.NotFound(w, r)
		}
	}))
	return srv, &createCalls, &patchCalls, &capturedPatchBody
}

// A credential that already exists in LiteLLM (created out of band, or left
// behind by a prior apply that dropped state) must be adopted on create
// instead of failing on the credential_name unique-constraint conflict, and
// the adopt PATCH must carry model_id so model-based credential resolution
// still applies (previously dropped - see
// https://github.com/BerriAI/litellm/pull/39745).
func TestResourceLiteLLMCredentialCreate_AdoptsOnConflict(t *testing.T) {
	srv, createCalls, patchCalls, patchBody := conflictServer(t, http.StatusOK, `{}`)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMCredential().Schema, map[string]interface{}{
		"credential_name":   "conflict-test",
		"model_id":          "model-1",
		"credential_info":   map[string]interface{}{"custom_llm_provider": "bedrock"},
		"credential_values": map[string]interface{}{"aws_access_key_id": "val"},
	})

	if err := resourceLiteLLMCredentialCreate(d, client); err != nil {
		t.Fatalf("expected create to adopt the existing credential, got error: %v", err)
	}
	if d.Id() != "conflict-test" {
		t.Fatalf("expected ID %q, got %q", "conflict-test", d.Id())
	}
	if got := atomic.LoadInt32(createCalls); got != 1 {
		t.Fatalf("expected exactly 1 POST /credentials call, got %d", got)
	}
	if got := atomic.LoadInt32(patchCalls); got != 1 {
		t.Fatalf("expected the conflict to trigger exactly 1 PATCH (adopt-and-update), got %d", got)
	}

	var sent map[string]interface{}
	if err := json.Unmarshal(*patchBody, &sent); err != nil {
		t.Fatalf("PATCH body was not valid JSON: %v (%s)", err, *patchBody)
	}
	if sent["credential_name"] != "conflict-test" {
		t.Errorf("PATCH body credential_name = %v, want conflict-test", sent["credential_name"])
	}
	if sent["model_id"] != "model-1" {
		t.Errorf("PATCH body model_id = %v, want model-1 (adoption must not drop model-based credential resolution)", sent["model_id"])
	}
	credInfo, _ := sent["credential_info"].(map[string]interface{})
	if credInfo["custom_llm_provider"] != "bedrock" {
		t.Errorf("PATCH body credential_info = %v, want custom_llm_provider=bedrock", sent["credential_info"])
	}
}

// If the adopt PATCH itself fails, create must not have set the resource ID
// for a credential this run doesn't own - otherwise Terraform taints the
// entry and the *next* apply destroys a credential nobody here created.
func TestResourceLiteLLMCredentialCreate_FailedAdoptDoesNotTaint(t *testing.T) {
	srv, createCalls, patchCalls, _ := conflictServer(t, http.StatusInternalServerError, `{"error":{"message":"Internal Server Error"}}`)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMCredential().Schema, map[string]interface{}{
		"credential_name":   "conflict-test",
		"credential_info":   map[string]interface{}{},
		"credential_values": map[string]interface{}{"key": "val"},
	})

	err := resourceLiteLLMCredentialCreate(d, client)
	if err == nil {
		t.Fatal("expected an error when the adopt PATCH fails, got nil")
	}
	if got := atomic.LoadInt32(createCalls); got != 1 {
		t.Fatalf("expected exactly 1 POST /credentials call, got %d", got)
	}
	if got := atomic.LoadInt32(patchCalls); got != 1 {
		t.Fatalf("expected exactly 1 PATCH attempt, got %d", got)
	}
	if d.Id() != "" {
		t.Fatalf("resource ID must stay empty after a failed adopt, got %q (a tainted entry would be destroyed on the next apply)", d.Id())
	}
}

// A non-conflict failure (a plain 500, for example) must return the original
// error and never attempt to adopt anything.
func TestResourceLiteLLMCredentialCreate_NonConflictErrorDoesNotAdopt(t *testing.T) {
	var createCalls, patchCalls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/credentials":
			atomic.AddInt32(&createCalls, 1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error":{"message":"Internal Server Error","type":"internal_server_error"}}`))
		case r.Method == http.MethodPatch:
			atomic.AddInt32(&patchCalls, 1)
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMCredential().Schema, map[string]interface{}{
		"credential_name":   "some-cred",
		"credential_info":   map[string]interface{}{},
		"credential_values": map[string]interface{}{"key": "val"},
	})

	err := resourceLiteLLMCredentialCreate(d, client)
	if err == nil {
		t.Fatal("expected an error for a non-conflict failure, got nil")
	}
	if got := atomic.LoadInt32(&patchCalls); got != 0 {
		t.Fatalf("expected no PATCH attempt for a non-conflict error, got %d", got)
	}
	if d.Id() != "" {
		t.Fatalf("resource ID must stay empty on a non-conflict failure, got %q", d.Id())
	}
}
