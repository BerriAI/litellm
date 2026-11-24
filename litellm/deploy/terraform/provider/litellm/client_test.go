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

	input := `{"credential_name":"azure-cred","credential_values":{"api_key":"sk-secret-123","config":{"region":"us-east-1"}}}`
	redacted := c.redactSensitiveData(input)

	if strings.Contains(redacted, "sk-secret-123") {
		t.Errorf("redacted output still contains sensitive api_key: %s", redacted)
	}
	if !strings.Contains(redacted, "[REDACTED]") {
		t.Errorf("redacted output missing [REDACTED] placeholder: %s", redacted)
	}
}

func TestPerPrincipalBudgetModelFamilyFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"budget_id":"b-123","models":["gpt-4*", "claude-3*"],"max_budget":100}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	result, err := client.sendRequest("POST", "/key/budget", map[string]interface{}{
		"models":     []string{"gpt-4*", "claude-3*"},
		"max_budget": 100,
	})

	if err != nil {
		t.Fatalf("unexpected error when configuring per-principal model family budget: %v", err)
	}

	if result["budget_id"] != "b-123" {
		t.Errorf("expected budget_id b-123, got %v", result["budget_id"])
	}
}