package litellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hashicorp/go-cty/cty"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func TestMCPServerReadDoesNotPersistServerEnv(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMMCPServer().Schema, map[string]interface{}{
		"server_name": "gh",
		"transport":   "stdio",
		"command":     "npx",
		"env": map[string]interface{}{
			"GITHUB_TOKEN": "from-config",
		},
	})
	d.SetId("srv-1")

	resp := &MCPServerResponse{
		ServerID:   "srv-1",
		ServerName: "gh",
		Transport:  "stdio",
		Command:    "npx",
		Env: map[string]string{
			"GITHUB_TOKEN": "raw-from-server",
			"DB_PASSWORD":  "leaked-secret",
		},
	}
	if err := updateSchemaFromResponse(d, resp); err != nil {
		t.Fatalf("updateSchemaFromResponse failed: %v", err)
	}

	got := d.Get("env").(map[string]interface{})
	if got["GITHUB_TOKEN"] != "from-config" {
		t.Fatalf("config env overwritten by server response: %v", got)
	}
	if _, leaked := got["DB_PASSWORD"]; leaked {
		t.Fatalf("server-returned env var persisted into state: %v", got)
	}
	if d.Get("server_name").(string) != "gh" {
		t.Fatalf("read did not populate non-sensitive fields")
	}
}

func TestMCPServerCreateSendsWriteOnlyCredentials(t *testing.T) {
	var created map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/mcp/server" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&created); err != nil {
			t.Errorf("decode create payload: %v", err)
		}
		w.Write([]byte(`{"server_id":"srv-1","server_name":"crm","url":"https://mcp.example.com","transport":"http","auth_type":"oauth2"}`))
	}))
	defer srv.Close()

	d := resourceLiteLLMMCPServer().Data(&terraform.InstanceState{
		Attributes: map[string]string{
			"server_name":              "crm",
			"url":                      "https://mcp.example.com",
			"transport":                "http",
			"auth_type":                "oauth2",
			"client_id":                "client-123",
			"client_secret_wo_version": "1",
		},
		RawConfig: cty.ObjectVal(map[string]cty.Value{
			"client_secret_wo": cty.StringVal("secret-only-in-config"),
		}),
	})

	if err := resourceLiteLLMMCPServerCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}
	creds, ok := created["credentials"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected credentials object in create payload, got %v", created)
	}
	if creds["client_id"] != "client-123" || creds["client_secret"] != "secret-only-in-config" {
		t.Fatalf("client_id/client_secret_wo not sent to the proxy, got %v", creds)
	}
	if _, present := creds["auth_value"]; present {
		t.Fatalf("auth_value must be omitted when auth_value_wo is unset, got %v", creds)
	}
	if d.Id() != "srv-1" {
		t.Fatalf("expected id srv-1, got %q", d.Id())
	}
}

func TestMCPServerUpdateSendsBearerAuthValue(t *testing.T) {
	var updated map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&updated); err != nil {
			t.Errorf("decode update payload: %v", err)
		}
		w.Write([]byte(`{"server_id":"srv-1","server_name":"crm","url":"https://mcp.example.com","transport":"http","auth_type":"bearer_token"}`))
	}))
	defer srv.Close()

	d := schema.TestResourceDataRaw(t, resourceLiteLLMMCPServer().Schema, map[string]interface{}{
		"server_name":           "crm",
		"url":                   "https://mcp.example.com",
		"transport":             "http",
		"auth_type":             "bearer_token",
		"auth_value_wo":         "rotated-token",
		"auth_value_wo_version": 2,
	})
	d.SetId("srv-1")

	if err := resourceLiteLLMMCPServerUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("update failed: %v", err)
	}
	creds, ok := updated["credentials"].(map[string]interface{})
	if !ok || creds["auth_value"] != "rotated-token" {
		t.Fatalf("auth_value_wo not sent on update, got %v", updated["credentials"])
	}
}

func TestMCPServerRequestOmitsCredentialsWhenNoneConfigured(t *testing.T) {
	d := schema.TestResourceDataRaw(t, resourceLiteLLMMCPServer().Schema, map[string]interface{}{
		"server_name": "plain",
		"url":         "https://mcp.example.com",
		"transport":   "http",
	})
	body, err := json.Marshal(buildMCPServerRequest(d))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), "credentials") {
		t.Fatalf("request must not carry an empty credentials object, got %s", body)
	}
}

func TestMCPServerSchemaWriteOnlyCredentialRules(t *testing.T) {
	base := map[string]interface{}{"server_name": "crm", "url": "https://mcp.example.com", "transport": "http"}
	cases := map[string]struct {
		extra   map[string]interface{}
		wantErr bool
	}{
		"oauth2 client secret with id and version": {
			extra:   map[string]interface{}{"auth_type": "oauth2", "client_id": "c", "client_secret_wo": "s", "client_secret_wo_version": 1},
			wantErr: false,
		},
		"client secret without client_id": {
			extra:   map[string]interface{}{"auth_type": "oauth2", "client_secret_wo": "s", "client_secret_wo_version": 1},
			wantErr: true,
		},
		"auth value without version": {
			extra:   map[string]interface{}{"auth_type": "bearer_token", "auth_value_wo": "t"},
			wantErr: true,
		},
		"legacy bearer still accepted": {
			extra:   map[string]interface{}{"auth_type": "bearer"},
			wantErr: false,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			raw := map[string]interface{}{}
			for k, v := range base {
				raw[k] = v
			}
			for k, v := range tc.extra {
				raw[k] = v
			}
			diags := resourceLiteLLMMCPServer().Validate(terraform.NewResourceConfigRaw(raw))
			if diags.HasError() != tc.wantErr {
				t.Fatalf("wantErr=%v, got diags %v", tc.wantErr, diags)
			}
		})
	}
}
