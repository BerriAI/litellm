package litellm

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// Regression test for https://github.com/BerriAI/litellm/issues/43019.
// The provider used to validate auth_type against ["none", "bearer", "basic"],
// but the LiteLLM API's MCPAuth enum (litellm/types/mcp.py) expects
// "bearer_token" among its canonical values. auth_type="bearer" passed
// provider validation and then failed with a 422 from the API, while
// auth_type="bearer_token" failed provider plan validation. The provider now
// accepts the API's canonical MCPAuth values verbatim.
func TestMCPServerAuthTypeValidationAcceptsCanonicalMCPAuthValues(t *testing.T) {
	vf := resourceLiteLLMMCPServer().Schema["auth_type"].ValidateFunc
	if vf == nil {
		t.Fatal("auth_type has no ValidateFunc")
	}

	valid := []string{
		"none",
		"api_key",
		"bearer_token",
		"basic",
		"authorization",
		"oauth2",
		"aws_sigv4",
		"token",
		"oauth2_token_exchange",
		"oauth2_id_jag",
		"true_passthrough",
		"oauth_delegate",
	}
	for _, v := range valid {
		if _, errs := vf(v, "auth_type"); len(errs) > 0 {
			t.Errorf("auth_type %q rejected, want accepted: %v", v, errs)
		}
	}
}

func TestMCPServerAuthTypeValidationRejectsNonCanonicalValues(t *testing.T) {
	vf := resourceLiteLLMMCPServer().Schema["auth_type"].ValidateFunc
	if vf == nil {
		t.Fatal("auth_type has no ValidateFunc")
	}

	// "bearer" is not a LiteLLM MCPAuth value; the API returns 422 for it, so
	// it must fail provider-side validation with a clear error instead of
	// reaching the API.
	invalid := []string{"bearer", "Bearer_Token", "BEARER", "jwt", "", " bearer_token"}
	for _, v := range invalid {
		if _, errs := vf(v, "auth_type"); len(errs) == 0 {
			t.Errorf("auth_type %q accepted, want rejected", v)
		}
	}
}

// The default must remain a valid value so existing configs without an
// explicit auth_type still plan cleanly.
func TestMCPServerAuthTypeDefaultIsValid(t *testing.T) {
	r := resourceLiteLLMMCPServer()
	def := r.Schema["auth_type"].Default.(string)
	if def != "none" {
		t.Fatalf("auth_type default = %q, want %q", def, "none")
	}
	if _, errs := r.Schema["auth_type"].ValidateFunc(def, "auth_type"); len(errs) > 0 {
		t.Errorf("auth_type default %q rejected: %v", def, errs)
	}
}

// auth_value carries the credential for auth types that need one (e.g.
// bearer_token, api_key). It must be optional and sensitive: hidden from plan
// output and never written back from API responses (the API redacts
// credentials on read, same as the sensitive "env" field).
func TestMCPServerAuthValueIsOptionalAndSensitive(t *testing.T) {
	f, ok := resourceLiteLLMMCPServer().Schema["auth_value"]
	if !ok {
		t.Fatal("schema has no auth_value field")
	}
	if !f.Optional {
		t.Error("auth_value should be Optional")
	}
	if !f.Sensitive {
		t.Error("auth_value must be Sensitive so the credential never appears in plan output")
	}
}

// The configured credential must reach the API as credentials.auth_value,
// matching the LiteLLM API's NewMCPServerRequest shape.
func TestMCPServerBuildRequestSendsAuthValueAsCredentials(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMMCPServer().Schema, map[string]interface{}{
		"server_name": "s",
		"url":         "https://example.com/mcp",
		"transport":   "http",
		"auth_type":   "bearer_token",
		"auth_value":  "secret-token",
	})
	req := buildMCPServerRequest(d)
	if req.Credentials == nil {
		t.Fatal("expected Credentials to be set when auth_value is configured")
	}
	if req.Credentials.AuthValue != "secret-token" {
		t.Errorf("Credentials.AuthValue = %q, want %q", req.Credentials.AuthValue, "secret-token")
	}
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	if !strings.Contains(string(body), `"credentials":{"auth_value":"secret-token"}`) {
		t.Errorf("request JSON missing credentials.auth_value, got: %s", body)
	}
}

// No credentials blob may be sent when auth_value is unset, so existing
// configs and auth_type="none" behave exactly as before.
func TestMCPServerBuildRequestOmitsCredentialsWhenAuthValueUnset(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMMCPServer().Schema, map[string]interface{}{
		"server_name": "s",
		"url":         "https://example.com/mcp",
		"transport":   "http",
		"auth_type":   "none",
	})
	req := buildMCPServerRequest(d)
	if req.Credentials != nil {
		t.Errorf("expected no Credentials when auth_value is unset, got %+v", req.Credentials)
	}
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	if strings.Contains(string(body), "credentials") {
		t.Errorf("request JSON must not contain credentials when auth_value is unset, got: %s", body)
	}
}
