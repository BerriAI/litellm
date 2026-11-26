package litellm

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
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

func TestProviderHeadersAreSentWithoutReplacingTheProvidersOwn(t *testing.T) {
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, Provider().Schema, map[string]interface{}{
		"api_base": srv.URL,
		"api_key":  "test-key",
		"headers": map[string]interface{}{
			"Proxy-Authorization": "Bearer proxy-token",
			"x-api-key":           "not-the-key",
		},
	})
	meta, err := providerConfigure(d)
	if err != nil {
		t.Fatalf("providerConfigure returned error: %v", err)
	}
	client := meta.(*Client)

	requests := map[string]func() error{
		"sendRequest": func() error {
			_, err := client.sendRequest("GET", "/whatever", nil)
			return err
		},
		"MakeRequest": func() error {
			resp, err := MakeRequest(client, "GET", "/whatever", nil)
			if err == nil {
				resp.Body.Close()
			}
			return err
		},
	}
	for name, send := range requests {
		t.Run(name, func(t *testing.T) {
			got = nil
			if err := send(); err != nil {
				t.Fatalf("request returned error: %v", err)
			}
			if v := got.Get("Proxy-Authorization"); v != "Bearer proxy-token" {
				t.Errorf("Proxy-Authorization = %q, want %q", v, "Bearer proxy-token")
			}
			if v := got.Get("x-api-key"); v != "test-key" {
				t.Errorf("x-api-key = %q, want the provider's api_key", v)
			}
		})
	}
}

func TestRedirectsMayNotLeaveTheHostOrDropTLS(t *testing.T) {
	redirect := func(from, to string) error {
		first := httptest.NewRequest("GET", from, nil)
		next := httptest.NewRequest("GET", to, nil)
		return refuseCrossOriginRedirect(next, []*http.Request{first})
	}

	allowed := map[string][2]string{
		"same host":             {"https://litellm.example.com/team/info", "https://litellm.example.com/v2/team/info"},
		"http to https upgrade": {"http://litellm.example.com/team/info", "https://litellm.example.com/team/info"},
	}
	for name, urls := range allowed {
		if err := redirect(urls[0], urls[1]); err != nil {
			t.Errorf("%s: redirect refused: %v", name, err)
		}
	}

	refused := map[string][2]string{
		"another host":            {"https://litellm.example.com/team/info", "https://login.example.net/"},
		"https to http downgrade": {"https://litellm.example.com/team/info", "http://litellm.example.com/team/info"},
	}
	for name, urls := range refused {
		if err := redirect(urls[0], urls[1]); err == nil {
			t.Errorf("%s: redirect followed, want it refused", name)
		}
	}
}

func TestProviderHeadersAreNotSentToARedirectsOtherHost(t *testing.T) {
	var reached bool
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
	}))
	defer other.Close()
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL+"/login", http.StatusFound)
	}))
	defer gateway.Close()

	client := NewClient(gateway.URL, "test-key", false)
	client.Headers = map[string]string{"Proxy-Authorization": "Bearer proxy-token"}

	if _, err := client.sendRequest("GET", "/team/info", nil); err == nil {
		t.Error("sendRequest followed the redirect, want an error")
	}
	if resp, err := MakeRequest(client, "GET", "/team/info", nil); err == nil {
		resp.Body.Close()
		t.Error("MakeRequest followed the redirect, want an error")
	}
	if reached {
		t.Error("the redirect's other host received a request")
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

	input := `{"data":[{"litellm_params":{"model":"gpt-4","api_key":"sk-deep-456","aws_secret_access_key":"aws-secret"}},{"credentials":{"client_private_key":"private-key-fixture","client_secret":"oauth-secret-fixture","id_jag_resource_client_secret":"id-jag-fixture"}}]}`
	got := c.redactSensitiveData(input)

	for _, leaked := range []string{"sk-deep-456", "aws-secret", "private-key-fixture", "oauth-secret-fixture", "id-jag-fixture"} {
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

func TestSendRequestRedactsCallbackDataInErrors(t *testing.T) {
	for _, tt := range []struct {
		name     string
		status   int
		body     string
		mustHide []string
	}{
		{
			name: "structured API error", status: http.StatusBadRequest,
			body:     `{"detail":{"callback_vars":{"custom_value":"fixture-private"},"provider_secret":"fixture-suffix","provider_secret_key":"fixture-key","provider_api_key":"fixture-api"}}`,
			mustHide: []string{"fixture-private", "fixture-suffix", "fixture-key", "fixture-api"},
		},
		{
			name: "non-JSON response", status: http.StatusOK,
			body:     `unexpected "provider_secret_key": "fixture-unstructured"`,
			mustHide: []string{"fixture-unstructured"},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				w.Write([]byte(tt.body))
			}))
			defer srv.Close()
			_, err := NewClient(srv.URL, "test-key", true).sendRequest("POST", "/key/update", nil)
			if err == nil {
				t.Fatal("expected an error")
			}
			for _, value := range tt.mustHide {
				if strings.Contains(err.Error(), value) {
					t.Errorf("error exposed %q: %v", value, err)
				}
			}
		})
	}
}
