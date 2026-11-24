package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/hashicorp/go-cty/cty"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func TestMCPServerOAuthRequestsAndReadBack(t *testing.T) {
	fields := map[string]interface{}{
		"issuer": "https://id.example.com", "authorization_url": "https://id.example.com/authorize",
		"token_url": "https://id.example.com/token", "registration_url": "https://id.example.com/register",
		"oauth2_flow": "client_credentials", "token_exchange_endpoint": "https://id.example.com/exchange",
		"audience": "api", "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
		"token_exchange_profile": "custom",
	}
	credentials := map[string]interface{}{
		"scopes": []interface{}{"read", "write"}, "token_endpoint_auth_method": "client_secret_basic",
		"upstream_resource": "https://api.example.com", "client_private_key_id": "key-1",
		"client_assertion_signing_alg": "RS256", "id_jag_resource_token_endpoint": "https://resource.example.com/token",
		"id_jag_resource": "resource", "id_jag_resource_client_id": "resource-client", "id_jag_subject_source": "refresh_token",
	}
	for _, operation := range []string{"create", "update"} {
		t.Run(operation, func(t *testing.T) {
			config := map[string]interface{}{"server_name": "crm", "url": "https://mcp.example.com", "transport": "http", "auth_type": "oauth2"}
			for k, v := range fields {
				config[k] = v
			}
			for k, v := range credentials {
				config[k] = v
			}
			config["client_private_key_wo"] = "private-key-fixture"
			config["client_private_key_wo_version"] = 1
			config["id_jag_resource_client_secret_wo"] = "resource-secret-fixture"
			config["id_jag_resource_client_secret_wo_version"] = 1
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var body map[string]interface{}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				for k, v := range fields {
					if body[k] != v {
						t.Errorf("%s request lost %s: %v", operation, k, body[k])
					}
				}
				got, _ := body["credentials"].(map[string]interface{})
				for k, v := range credentials {
					if !reflect.DeepEqual(got[k], v) {
						t.Errorf("credential %s: got %v want %v", k, got[k], v)
					}
				}
				if got["client_private_key"] != "private-key-fixture" || got["id_jag_resource_client_secret"] != "resource-secret-fixture" {
					t.Error("write-only credentials missing")
				}
				body["server_id"] = "srv-1"
				body["credentials"] = map[string]interface{}{"scopes": []string{"read", "write"}, "upstream_resource": "https://api.example.com", "client_private_key": "must-not-persist", "id_jag_resource_client_secret": "must-not-persist"}
				json.NewEncoder(w).Encode(body)
			}))
			defer srv.Close()
			res := resourceLiteLLMMCPServer()
			delete(config, "client_private_key_wo")
			delete(config, "id_jag_resource_client_secret_wo")
			var prior *terraform.InstanceState
			if operation == "update" {
				initial := schema.TestResourceDataRaw(t, res.Schema, map[string]interface{}{
					"server_name": "crm", "url": "https://mcp.example.com", "transport": "http", "auth_type": "oauth2",
				})
				initial.SetId("srv-1")
				prior = initial.State()
			}
			diff, err := res.Diff(context.Background(), prior, terraform.NewResourceConfigRaw(config), nil)
			if err != nil {
				t.Fatal(err)
			}
			diff.RawConfig = cty.ObjectVal(map[string]cty.Value{
				"client_private_key_wo":            cty.StringVal("private-key-fixture"),
				"id_jag_resource_client_secret_wo": cty.StringVal("resource-secret-fixture"),
			})
			d, err := schema.InternalMap(res.Schema).Data(prior, diff)
			if err != nil {
				t.Fatal(err)
			}
			client := NewClient(srv.URL, "test-key", true)
			if operation == "create" {
				err = resourceLiteLLMMCPServerCreate(d, client)
			} else {
				d.SetId("srv-1")
				err = resourceLiteLLMMCPServerUpdate(d, client)
			}
			if err != nil {
				t.Fatal(err)
			}
			if calls == 0 {
				t.Fatal("no request received")
			}
			for k, v := range fields {
				if d.Get(k) != v {
					t.Errorf("read-back lost %s", k)
				}
			}
			if !reflect.DeepEqual(d.Get("scopes"), credentials["scopes"]) || d.Get("upstream_resource") != credentials["upstream_resource"] {
				t.Fatal("credential read-back missing")
			}
			state, _ := json.Marshal(d.State().Attributes)
			if strings.Contains(string(state), "private-key-fixture") || strings.Contains(string(state), "resource-secret-fixture") || strings.Contains(string(state), "must-not-persist") {
				t.Fatal("secret persisted in state")
			}
		})
	}
}

func TestMCPServerOAuthUpdatePreservesOmittedAndClearsExplicit(t *testing.T) {
	for _, changed := range []string{"issuer", "url", "auth_type", "description", "omit_token_url", "omit_scopes", "scopes", "client_secret_wo_version", "token_endpoint_auth_method", "client_private_key_id", "client_assertion_signing_alg", "id_jag_resource_token_endpoint", "id_jag_resource", "id_jag_resource_client_id", "id_jag_subject_source"} {
		t.Run(changed, func(t *testing.T) {
			res := resourceLiteLLMMCPServer()
			config := map[string]interface{}{"server_name": "crm", "url": "https://mcp.example.com", "transport": "http", "auth_type": "oauth2", "issuer": "https://id.example.com", "token_url": "https://id.example.com/token", "authorization_url": "https://id.example.com/auth", "registration_url": "https://id.example.com/register", "client_id": "client", "scopes": []interface{}{"read"}, "upstream_resource": "api"}
			credentialStrings := map[string]interface{}{
				"upstream_resource": "api", "token_endpoint_auth_method": "client_secret_basic",
				"client_private_key_id": "key", "client_assertion_signing_alg": "RS256",
				"id_jag_resource_token_endpoint": "https://resource.example.com/token",
				"id_jag_resource":                "api", "id_jag_resource_client_id": "resource-client",
				"id_jag_subject_source": "refresh_token",
			}
			for field, value := range credentialStrings {
				config[field] = value
			}
			config["oauth2_flow"] = "client_credentials"
			config["audience"] = "unmanaged-audience"
			config["client_secret_wo_version"] = 1
			config["client_private_key_wo_version"] = 1
			prior := schema.TestResourceDataRaw(t, res.Schema, config)
			prior.SetId("srv-1")
			delete(config, "audience")
			if changed == "issuer" || changed == "url" {
				config[changed] = "https://new.example.com"
			} else if changed == "auth_type" {
				config[changed] = "oauth2_id_jag"
			} else if changed == "description" {
				config[changed] = "new description"
			} else if strings.HasPrefix(changed, "omit_") {
				delete(config, strings.TrimPrefix(changed, "omit_"))
				config["description"] = "new description"
			} else if changed == "scopes" {
				config[changed] = []interface{}{}
			} else if changed == "client_secret_wo_version" {
				config[changed] = 2
			} else {
				delete(config, changed)
			}
			diff, err := res.Diff(context.Background(), prior.State(), terraform.NewResourceConfigRaw(config), nil)
			if err != nil {
				t.Fatal(err)
			}
			rawConfig := map[string]cty.Value{
				"client_secret_wo":      cty.StringVal("client-secret-fixture"),
				"client_private_key_wo": cty.StringVal("private-key-fixture"),
			}
			for field, value := range config {
				if s, ok := value.(string); ok {
					rawConfig[field] = cty.StringVal(s)
				}
			}
			if changed != "omit_scopes" && changed != "scopes" {
				rawConfig["scopes"] = cty.ListVal([]cty.Value{cty.StringVal("read")})
			}
			diff.RawConfig = cty.ObjectVal(rawConfig)
			d, err := schema.InternalMap(res.Schema).Data(prior.State(), diff)
			if err != nil {
				t.Fatal(err)
			}
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var body map[string]interface{}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				creds, _ := body["credentials"].(map[string]interface{})
				if changed == "client_secret_wo_version" || changed == "auth_type" {
					if creds["client_secret"] != "client-secret-fixture" || creds["client_id"] != "client" {
						t.Error("secret rotation must send the OAuth client and its updated secret")
					}
				} else {
					if _, clientPresent := creds["client_id"]; clientPresent {
						t.Error("update replayed an unchanged OAuth client")
					}
					if _, secretPresent := creds["client_secret"]; secretPresent {
						t.Error("update replayed an unchanged OAuth secret")
					}
				}
				if changed == "url" || changed == "issuer" || changed == "auth_type" {
					if body["preserve_explicit_oauth_settings"] != true ||
						body["issuer"] != config["issuer"] || body["oauth2_flow"] != "client_credentials" ||
						body["token_url"] != config["token_url"] {
						t.Errorf("%s edit dropped explicitly configured OAuth settings", changed)
					}
					if _, present := body["audience"]; present {
						t.Errorf("%s edit replayed unmanaged OAuth settings", changed)
					}
					if creds["token_endpoint_auth_method"] != "client_secret_basic" {
						t.Errorf("%s edit dropped the configured token authentication method", changed)
					}
				}
				if changed == "auth_type" {
					scopes, ok := creds["scopes"].([]interface{})
					if !ok || len(scopes) != 1 || scopes[0] != "read" ||
						creds["client_private_key"] != "private-key-fixture" {
						t.Error("auth type switch dropped configured scopes or private key")
					}
				} else if _, present := creds["client_private_key"]; present {
					t.Error("update replayed an unchanged private key")
				}
				if changed == "issuer" && body["issuer"] != config["issuer"] {
					t.Error("issuer edit did not send the new issuer")
				}
				if changed == "omit_token_url" {
					if _, present := body["token_url"]; present {
						t.Error("update changed an omitted token URL")
					}
				}
				if changed == "omit_scopes" {
					if _, present := creds["scopes"]; present {
						t.Error("update changed omitted scopes")
					}
				}
				if _, clearingCredential := credentialStrings[changed]; clearingCredential {
					if value, present := creds[changed]; !present || value != nil {
						t.Errorf("removed credential %s must be explicit null", changed)
					}
				}
				if changed == "scopes" {
					if v, ok := creds["scopes"].([]interface{}); !ok || len(v) != 0 {
						t.Error("scopes were not cleared")
					}
				}
				body["server_id"] = "srv-1"
				json.NewEncoder(w).Encode(body)
			}))
			defer srv.Close()
			if err := resourceLiteLLMMCPServerUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
				t.Fatal(err)
			}
			if calls != 1 {
				t.Fatalf("got %d PUTs, want 1", calls)
			}
		})
	}
}

func TestMCPServerOAuthClearsWriteOnlySecrets(t *testing.T) {
	res := resourceLiteLLMMCPServer()
	config := map[string]interface{}{"server_name": "crm", "url": "https://mcp.example.com", "transport": "http", "auth_type": "oauth2", "client_id": "client"}
	prior := schema.TestResourceDataRaw(t, res.Schema, config)
	prior.SetId("srv-1")
	prior.Set("client_private_key_wo_version", 1)
	prior.Set("id_jag_resource_client_secret_wo_version", 1)
	diff, err := res.Diff(context.Background(), prior.State(), terraform.NewResourceConfigRaw(config), nil)
	if err != nil {
		t.Fatal(err)
	}
	d, err := schema.InternalMap(res.Schema).Data(prior.State(), diff)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		credentials, _ := body["credentials"].(map[string]interface{})
		for _, field := range []string{"client_private_key", "id_jag_resource_client_secret"} {
			if value, present := credentials[field]; !present || value != nil {
				t.Errorf("removed write-only credential %s must be explicit null", field)
			}
		}
		body["server_id"] = "srv-1"
		json.NewEncoder(w).Encode(body)
	}))
	defer srv.Close()
	if err := resourceLiteLLMMCPServerUpdate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatal(err)
	}
}

func TestMCPServerDeleteAcceptsFullSuccessRange(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		wantErr    bool
	}{
		{name: "200 OK", statusCode: http.StatusOK, wantErr: false},
		{name: "202 Accepted", statusCode: http.StatusAccepted, wantErr: false},
		{name: "404 Not Found", statusCode: http.StatusNotFound, wantErr: true},
		{name: "500 Internal Server Error", statusCode: http.StatusInternalServerError, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
			}))
			defer srv.Close()

			d := schema.TestResourceDataRaw(t, resourceLiteLLMMCPServer().Schema, map[string]interface{}{
				"server_name": "gh",
				"transport":   "stdio",
				"command":     "npx",
			})
			d.SetId("srv-1")

			client := NewClient(srv.URL, "test-key", true)
			err := resourceLiteLLMMCPServerDelete(d, client)

			if tt.wantErr {
				if err == nil {
					t.Fatalf("resourceLiteLLMMCPServerDelete returned no error for status %d", tt.statusCode)
				}
				return
			}

			if err != nil {
				t.Fatalf("resourceLiteLLMMCPServerDelete returned unexpected error: %v", err)
			}
			if d.Id() != "" {
				t.Errorf("resource ID not cleared after successful delete, got %q", d.Id())
			}
		})
	}
}

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
		"private key without version":             {extra: map[string]interface{}{"client_private_key_wo": "key"}, wantErr: true},
		"resource client secret without version":  {extra: map[string]interface{}{"id_jag_resource_client_secret_wo": "secret"}, wantErr: true},
		"invalid flow":                            {extra: map[string]interface{}{"oauth2_flow": "implicit"}, wantErr: true},
		"unsupported token authentication method": {extra: map[string]interface{}{"token_endpoint_auth_method": "private_key_jwt"}, wantErr: true},
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

func TestMCPServerReadProducesStablePlan(t *testing.T) {
	const responseWithoutModeledMCPInfo = `{"alias":"lit9695_proof","approval_status":"active","args":[],"auth_type":"none","command":null,"created_at":"2026-10-15T22:48:21.530000Z","created_by":"default_user_id","credentials":null,"description":null,"env":{},"mcp_access_groups":[],"mcp_info":{"is_public":false,"is_public_explicit":false},"server_id":"dc577aa3-22ac-4fdf-a56b-f6c40e35c389","server_name":"lit9695_proof","status":"unknown","transport":"http","updated_at":"2026-10-15T22:48:21.530000Z","updated_by":"default_user_id","url":"https://mcp.example.com/mcp"}`
	responseWithLogoURL := strings.Replace(
		responseWithoutModeledMCPInfo,
		`"mcp_info":{"is_public":false,"is_public_explicit":false}`,
		`"mcp_info":{"is_public":false,"is_public_explicit":false,"logo_url":"https://x/logo.png"}`,
		1,
	)
	if responseWithLogoURL == responseWithoutModeledMCPInfo {
		t.Fatal("failed to add logo_url to MCP info response")
	}

	cases := []struct {
		name               string
		config             map[string]interface{}
		response           string
		wantEmptyDiff      bool
		wantChangedKey     string
		wantAlias          string
		wantSpecVersion    string
		wantMCPInfoLogoURL string
	}{
		{
			name: "minimal config",
			config: map[string]interface{}{
				"server_name": "lit9695_proof",
				"url":         "https://mcp.example.com/mcp",
				"transport":   "http",
			},
			response:        responseWithoutModeledMCPInfo,
			wantEmptyDiff:   true,
			wantAlias:       "lit9695_proof",
			wantSpecVersion: "2024-11-05",
		},
		{
			name: "configured spec version",
			config: map[string]interface{}{
				"server_name":  "lit9695_proof",
				"url":          "https://mcp.example.com/mcp",
				"transport":    "http",
				"spec_version": "2025-03-26",
			},
			response:        responseWithoutModeledMCPInfo,
			wantEmptyDiff:   true,
			wantSpecVersion: "2025-03-26",
		},
		{
			name: "configured alias drift remains visible",
			config: map[string]interface{}{
				"server_name": "lit9695_proof",
				"alias":       "custom",
				"url":         "https://mcp.example.com/mcp",
				"transport":   "http",
			},
			response:       responseWithoutModeledMCPInfo,
			wantChangedKey: "alias",
		},
		{
			name: "configured MCP info drift remains visible",
			config: map[string]interface{}{
				"server_name": "lit9695_proof",
				"url":         "https://mcp.example.com/mcp",
				"transport":   "http",
				"mcp_info": []interface{}{
					map[string]interface{}{"logo_url": "https://x/logo.png"},
				},
			},
			response:       responseWithoutModeledMCPInfo,
			wantChangedKey: "mcp_info",
		},
		{
			name: "modeled MCP info is retained",
			config: map[string]interface{}{
				"server_name": "lit9695_proof",
				"url":         "https://mcp.example.com/mcp",
				"transport":   "http",
				"mcp_info": []interface{}{
					map[string]interface{}{"logo_url": "https://x/logo.png"},
				},
			},
			response:           responseWithLogoURL,
			wantEmptyDiff:      true,
			wantMCPInfoLogoURL: "https://x/logo.png",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost && r.Method != http.MethodGet {
					t.Errorf("unexpected request method: %s", r.Method)
					w.WriteHeader(http.StatusNotFound)
					return
				}
				if r.URL.Path != "/v1/mcp/server" && r.URL.Path != "/v1/mcp/server/dc577aa3-22ac-4fdf-a56b-f6c40e35c389" {
					t.Errorf("unexpected request path: %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(tc.response))
			}))
			defer server.Close()

			resource := resourceLiteLLMMCPServer()
			d := schema.TestResourceDataRaw(t, resource.Schema, tc.config)
			client := NewClient(server.URL, "test-key", true)
			if err := resourceLiteLLMMCPServerCreate(d, client); err != nil {
				t.Fatalf("create failed: %v", err)
			}
			if err := resourceLiteLLMMCPServerRead(d, client); err != nil {
				t.Fatalf("read failed: %v", err)
			}

			diff, err := resource.Diff(context.Background(), d.State(), terraform.NewResourceConfigRaw(tc.config), nil)
			if err != nil {
				t.Fatalf("plan diff failed: %v", err)
			}
			if tc.wantEmptyDiff {
				if diff != nil && !diff.Empty() {
					t.Fatalf("expected empty plan diff, got %v", diff.Attributes)
				}
				if diff != nil {
					for key := range diff.Attributes {
						if key == "alias" || key == "spec_version" || strings.HasPrefix(key, "mcp_info.") {
							t.Fatalf("unexpected diff on %s", key)
						}
					}
				}
			} else {
				if diff == nil || diff.Empty() {
					t.Fatal("expected a plan diff")
				}
				hasExpectedDiff := false
				for key := range diff.Attributes {
					hasExpectedDiff = hasExpectedDiff || key == tc.wantChangedKey ||
						(tc.wantChangedKey == "mcp_info" && strings.HasPrefix(key, "mcp_info."))
				}
				if !hasExpectedDiff {
					t.Fatalf("expected a plan diff on %s, got %v", tc.wantChangedKey, diff.Attributes)
				}
			}

			if tc.wantAlias != "" && d.Get("alias") != tc.wantAlias {
				t.Fatalf("alias = %v, want %s", d.Get("alias"), tc.wantAlias)
			}
			if tc.wantSpecVersion != "" && d.Get("spec_version") != tc.wantSpecVersion {
				t.Fatalf("spec_version = %v, want %s", d.Get("spec_version"), tc.wantSpecVersion)
			}
			if tc.wantMCPInfoLogoURL != "" {
				mcpInfo := d.Get("mcp_info").([]interface{})
				if len(mcpInfo) != 1 {
					t.Fatalf("mcp_info has %d blocks, want 1", len(mcpInfo))
				}
				gotLogoURL := mcpInfo[0].(map[string]interface{})["logo_url"]
				if gotLogoURL != tc.wantMCPInfoLogoURL {
					t.Fatalf("mcp_info logo_url = %v, want %s", gotLogoURL, tc.wantMCPInfoLogoURL)
				}
			}
		})
	}
}
