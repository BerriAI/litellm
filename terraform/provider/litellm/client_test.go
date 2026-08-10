package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testClient(apiBase, apiKey string, insecure bool, headers map[string]string) *Client {
	return NewClient(ProviderConfig{
		APIBase:            apiBase,
		APIKey:             apiKey,
		InsecureSkipVerify: insecure,
		CustomHeaders:      headers,
	})
}

func TestSendRequestCustomHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Proxy-Auth"); got != "proxy-token" {
			t.Errorf("X-Proxy-Auth = %q, want %q", got, "proxy-token")
		}
		if got := r.Header.Get("X-Request-Source"); got != "terraform" {
			t.Errorf("X-Request-Source = %q, want %q", got, "terraform")
		}
		if got := r.Header.Get("x-api-key"); got != "sk-real-key" {
			t.Errorf("x-api-key = %q, want configured API key (must not be overridden)", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
	}))
	defer srv.Close()

	client := testClient(srv.URL, "sk-real-key", false, map[string]string{
		"X-Proxy-Auth":     "proxy-token",
		"X-Request-Source": "terraform",
		"x-api-key":        "should-not-win",
	})

	if _, err := client.sendRequest("GET", "/health", nil); err != nil {
		t.Fatalf("sendRequest: %v", err)
	}
}

func TestMakeRequestCustomHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Proxy-Auth"); got != "proxy-token" {
			t.Errorf("X-Proxy-Auth = %q, want %q", got, "proxy-token")
		}
		if got := r.Header.Get("x-api-key"); got != "sk-real-key" {
			t.Errorf("x-api-key = %q, want configured API key (must not be overridden)", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
	}))
	defer srv.Close()

	client := testClient(srv.URL, "sk-real-key", false, map[string]string{
		"X-Proxy-Auth": "proxy-token",
		"x-api-key":    "should-not-win",
	})

	resp, err := MakeRequest(client, "GET", "/health", nil)
	if err != nil {
		t.Fatalf("MakeRequest: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

func TestRedactSensitiveDataNestedCredentialValues(t *testing.T) {
	c := testClient("http://localhost:4000", "sk-test", false, nil)

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
	c := testClient("http://localhost:4000", "sk-test", false, nil)

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
	c := testClient("http://localhost:4000", "sk-test", false, nil)

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
	c := testClient("http://localhost:4000", "sk-test", false, nil)

	input := `error before "api_key": "sk-fallback-000" after`
	got := c.redactSensitiveData(input)

	if strings.Contains(got, "sk-fallback-000") {
		t.Errorf("fallback redaction leaked secret: %s", got)
	}
	if !strings.Contains(got, "[REDACTED]") {
		t.Errorf("fallback redaction did not redact: %s", got)
	}
}
