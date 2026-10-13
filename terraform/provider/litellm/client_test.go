package litellm

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRedactSensitiveDataNestedCredentialValues(t *testing.T) {
	c := NewClient("http://localhost:4000", "sk-test", false)

	input := `{"credential_name":"azure-cred","credential_values":{"api_key":"sk-secret-123","config":{"region":"us-east-1","client_secret":"nested-secret"}}}`
	got := c.redactSensitiveData(input)

	for _, leaked := range []string{"sk-secret-123", "us-east-1", "nested-secret"} {
		if strings.Contains(got, leaked) {
			t.Errorf("redacted output leaked %q: %s", leaked, got)
		}
	}
	if !strings.Contains(got, `"credential_values":"[REDACTED]"`) {
		t.Errorf("credential_values not redacted: %s", got)
	}
	if !strings.Contains(got, `"credential_name":"azure-cred"`) {
		t.Errorf("non-sensitive field mangled: %s", got)
	}
}

func TestRedactSensitiveDataDeeplyNestedSensitiveKeys(t *testing.T) {
	c := NewClient("http://localhost:4000", "sk-test", false)

	input := `{"data":[{"litellm_params":{"model":"gpt-4","api_key":"sk-deep-456","aws_secret_access_key":"aws-secret"}}]}`
	got := c.redactSensitiveData(input)

	for _, leaked := range []string{"sk-deep-456", "aws-secret"} {
		if strings.Contains(got, leaked) {
			t.Errorf("redacted output leaked %q: %s", leaked, got)
		}
	}
	if !strings.Contains(got, `"model":"gpt-4"`) {
		t.Errorf("non-sensitive field mangled: %s", got)
	}
}

func TestRedactSensitiveDataTopLevelStringFields(t *testing.T) {
	c := NewClient("http://localhost:4000", "sk-test", false)

	input := `{"model_api_key":"sk-top-789","vertex_credentials":"{\"type\":\"service_account\"}","team_alias":"eng"}`
	got := c.redactSensitiveData(input)

	for _, leaked := range []string{"sk-top-789", "service_account"} {
		if strings.Contains(got, leaked) {
			t.Errorf("redacted output leaked %q: %s", leaked, got)
		}
	}
	if !strings.Contains(got, `"team_alias":"eng"`) {
		t.Errorf("non-sensitive field mangled: %s", got)
	}
}

func TestRedactSensitiveDataNonJSONFallback(t *testing.T) {
	c := NewClient("http://localhost:4000", "sk-test", false)

	input := `error before "api_key": "sk-fallback-000" after`
	got := c.redactSensitiveData(input)

	if strings.Contains(got, "sk-fallback-000") {
		t.Errorf("fallback redaction leaked secret: %s", got)
	}
	if !strings.Contains(got, "[REDACTED]") {
		t.Errorf("fallback redaction did not redact: %s", got)
	}
}

func TestGetKeyReturnsNilForDeletedStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"info":{"key_alias":"deleted-key-repro","status":"deleted","deleted_at":"2026-10-12T00:00:00Z"}}`)
	}))
	defer server.Close()

	c := NewClient(server.URL, "sk-test", false)
	key, err := c.GetKey("sk-deleted")
	if err != nil {
		t.Fatalf("GetKey returned error: %v", err)
	}
	if key != nil {
		t.Errorf("GetKey returned a key for status=deleted, want nil")
	}
}
