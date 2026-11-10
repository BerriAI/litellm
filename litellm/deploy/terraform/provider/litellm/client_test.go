package litellm

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSendRequestAcceptsFullSuccessRange(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		wantErr    bool
		wantValue  string
	}{
		{name: "200 OK", statusCode: http.StatusOK, wantErr: false, wantValue: "ok"},
		{name: "201 Created", statusCode: http.StatusCreated, wantErr: false, wantValue: "created"},
		{name: "202 Accepted", statusCode: http.StatusAccepted, wantErr: false, wantValue: "accepted"},
		{name: "400 Bad Request", statusCode: http.StatusBadRequest, wantErr: true},
		{name: "404 Not Found", statusCode: http.StatusNotFound, wantErr: true},
		{name: "500 Internal Server Error", statusCode: http.StatusInternalServerError, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.statusCode)
				w.Write([]byte(`{"value":"` + tt.wantValue + `"}`))
			}))
			defer srv.Close()

			client := NewClient(srv.URL, "test-key", true)
			result, err := client.sendRequest("POST", "/whatever", map[string]string{"foo": "bar"})

			if tt.wantErr {
				if err == nil {
					t.Fatalf("sendRequest returned no error for status %d", tt.statusCode)
				}
				var apiErr *apiError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error is not *apiError: %v", err)
				}
				if apiErr.StatusCode != tt.statusCode {
					t.Errorf("apiErr.StatusCode = %d, want %d", apiErr.StatusCode, tt.statusCode)
				}
				if tt.statusCode == http.StatusNotFound && !isNotFound(err) {
					t.Errorf("isNotFound(err) = false for 404, want true")
				}
				return
			}

			if err != nil {
				t.Fatalf("sendRequest returned unexpected error: %v", err)
			}
			if result["value"] != tt.wantValue {
				t.Errorf("result[value] = %v, want %q", result["value"], tt.wantValue)
			}
		})
	}
}

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
