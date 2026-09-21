package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

// resourceDataWithChange builds a ResourceData carrying a real diff between
// prior state and new config, so d.GetChange reflects true old/new values.
// schema.TestResourceDataRaw diffs against a nil prior state, which collapses
// GetChange's old side to the zero value and can't exercise this.
func resourceDataWithChange(t *testing.T, oldAttrs map[string]string, newRaw map[string]interface{}) *schema.ResourceData {
	t.Helper()

	sm := schema.InternalMap(resourceLiteLLMJWTKeyMapping().Schema)
	state := &terraform.InstanceState{ID: oldAttrs["id"], Attributes: oldAttrs}
	config := terraform.NewResourceConfigRaw(newRaw)

	diff, err := sm.Diff(context.Background(), state, config, nil, nil, true)
	if err != nil {
		t.Fatalf("diff: %v", err)
	}
	d, err := sm.Data(state, diff)
	if err != nil {
		t.Fatalf("data: %v", err)
	}
	return d
}

type jwtKeyMappingCall struct {
	Method string
	Path   string
	Query  string
	Body   map[string]interface{}
}

func jwtKeyMappingTestServer(t *testing.T, mapping JWTKeyMappingResponse) (*httptest.Server, *[]jwtKeyMappingCall) {
	t.Helper()

	calls := make([]jwtKeyMappingCall, 0)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := map[string]interface{}{}
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		calls = append(calls, jwtKeyMappingCall{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Body: body})

		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/jwt/key/mapping/delete":
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "success"})
		default:
			_ = json.NewEncoder(w).Encode(mapping)
		}
	}))

	return srv, &calls
}

func jwtKeyMappingFixture() JWTKeyMappingResponse {
	return JWTKeyMappingResponse{
		ID:            "map-abc-123",
		JWTClaimName:  "client_id",
		JWTClaimValue: "dev-alice",
		Description:   "dev-alice",
		IsActive:      true,
		CreatedAt:     "2026-08-06T10:00:00Z",
		UpdatedAt:     "2026-08-06T11:00:00Z",
		CreatedBy:     "admin",
		UpdatedBy:     "admin",
	}
}

func TestJWTKeyMappingCreateSendsClaimAndKey(t *testing.T) {
	srv, calls := jwtKeyMappingTestServer(t, jwtKeyMappingFixture())
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-abc123",
		"description":     "dev-alice",
		"is_active":       true,
	})

	if err := resourceLiteLLMJWTKeyMappingCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if d.Id() != "map-abc-123" {
		t.Fatalf("expected id from the API response, got %q", d.Id())
	}

	create := (*calls)[0]
	if create.Method != "POST" || create.Path != "/jwt/key/mapping/new" {
		t.Fatalf("expected POST /jwt/key/mapping/new, got %s %s", create.Method, create.Path)
	}
	if create.Body["jwt_claim_name"] != "client_id" || create.Body["jwt_claim_value"] != "dev-alice" {
		t.Fatalf("claim fields not sent: %v", create.Body)
	}
	if create.Body["key"] != "sk-abc123" {
		t.Fatalf("virtual key not sent: %v", create.Body["key"])
	}
	if create.Body["description"] != "dev-alice" {
		t.Fatalf("description not sent: %v", create.Body["description"])
	}
	if _, sent := create.Body["is_active"]; sent {
		t.Fatalf("is_active is not accepted by /jwt/key/mapping/new but was sent: %v", create.Body)
	}

	for _, call := range (*calls)[1:] {
		if call.Path == "/jwt/key/mapping/update" {
			t.Fatalf("an active mapping must not trigger a follow-up update")
		}
	}
}

func TestJWTKeyMappingCreateOmitsEmptyDescription(t *testing.T) {
	srv, calls := jwtKeyMappingTestServer(t, jwtKeyMappingFixture())
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-abc123",
		"is_active":       true,
	})

	if err := resourceLiteLLMJWTKeyMappingCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	if _, sent := (*calls)[0].Body["description"]; sent {
		t.Fatalf("unset description should be omitted: %v", (*calls)[0].Body)
	}
}

func TestJWTKeyMappingCreateDeactivatesWhenNotActive(t *testing.T) {
	mapping := jwtKeyMappingFixture()
	mapping.IsActive = false
	srv, calls := jwtKeyMappingTestServer(t, mapping)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-abc123",
		"is_active":       false,
	})

	if err := resourceLiteLLMJWTKeyMappingCreate(d, client); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	var update *jwtKeyMappingCall
	for i := range *calls {
		if (*calls)[i].Path == "/jwt/key/mapping/update" {
			update = &(*calls)[i]
			break
		}
	}
	if update == nil {
		t.Fatal("expected a follow-up update, since the create endpoint always starts a mapping active")
	}
	if update.Body["id"] != "map-abc-123" {
		t.Fatalf("update must target the new mapping, got %v", update.Body["id"])
	}
	if update.Body["is_active"] != false {
		t.Fatalf("expected is_active false in the follow-up update, got %v", update.Body["is_active"])
	}
	if d.Get("is_active").(bool) {
		t.Fatal("state should reflect the inactive mapping after create")
	}
}

func TestJWTKeyMappingCreateDeletesMappingWhenDeactivationFails(t *testing.T) {
	// Regression test: the create endpoint has no is_active field and always
	// activates the mapping, so a failed deactivation used to leave that
	// mapping active and unmanaged indefinitely. It must be deleted instead.
	calls := make([]jwtKeyMappingCall, 0)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := map[string]interface{}{}
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		calls = append(calls, jwtKeyMappingCall{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Body: body})

		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/jwt/key/mapping/new":
			_ = json.NewEncoder(w).Encode(jwtKeyMappingFixture())
		case "/jwt/key/mapping/update":
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{"detail": "proxy unavailable"})
		case "/jwt/key/mapping/delete":
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "success"})
		default:
			t.Fatalf("unexpected request to %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-abc123",
		"is_active":       false,
	})

	err := resourceLiteLLMJWTKeyMappingCreate(d, client)
	if err == nil {
		t.Fatal("expected the failed deactivation to surface as an error")
	}
	if !strings.Contains(err.Error(), "deleted instead") {
		t.Fatalf("expected the error to explain the mapping was deleted, got %v", err)
	}

	deleteCalls := 0
	for _, c := range calls {
		if c.Path == "/jwt/key/mapping/delete" {
			deleteCalls++
			if c.Body["id"] != "map-abc-123" {
				t.Fatalf("delete must target the mapping that could not be deactivated, got %v", c.Body["id"])
			}
		}
	}
	if deleteCalls != 1 {
		t.Fatalf("expected exactly one cleanup delete call, got %d", deleteCalls)
	}

	if d.Id() != "" {
		t.Fatalf("a successfully deleted mapping must not remain in state, got id %q", d.Id())
	}
}

func TestJWTKeyMappingCreateReportsWhenDeactivationAndDeleteBothFail(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/jwt/key/mapping/new":
			_ = json.NewEncoder(w).Encode(jwtKeyMappingFixture())
		default:
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{"detail": "proxy unavailable"})
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-abc123",
		"is_active":       false,
	})

	err := resourceLiteLLMJWTKeyMappingCreate(d, client)
	if err == nil {
		t.Fatal("expected an error when both deactivation and the cleanup delete fail")
	}
	if !strings.Contains(err.Error(), "remove it manually") {
		t.Fatalf("expected the error to demand manual cleanup, got %v", err)
	}

	// The mapping is still active on the proxy since neither call succeeded, so
	// the id must stay in state: the next apply taints and retries the delete,
	// rather than Terraform losing track of a live, active mapping entirely.
	if d.Id() != "map-abc-123" {
		t.Fatalf("expected the id to remain in state so a retry can find it, got %q", d.Id())
	}
}

func TestJWTKeyMappingUpdateRevertsDescriptionAndIsActiveWhenTheRecoveryReadAlsoFails(t *testing.T) {
	// Regression test: on a failed update, only `key` was being reverted
	// before Read ran. If Read itself then failed too (network blip, proxy
	// hiccup), description/is_active kept the rejected, never-applied values,
	// and Terraform could persist them as if the update had succeeded.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/jwt/key/mapping/update":
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"detail": "rejected"})
		case "/jwt/key/mapping/info":
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{"detail": "proxy unavailable"})
		default:
			t.Fatalf("unexpected request to %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)

	d := resourceDataWithChange(t,
		map[string]string{
			"id":              "map-abc-123",
			"jwt_claim_name":  "client_id",
			"jwt_claim_value": "dev-alice",
			"key":             "sk-old-key-0000000000",
			"description":     "old description",
			"is_active":       "true",
		},
		map[string]interface{}{
			"jwt_claim_name":  "client_id",
			"jwt_claim_value": "dev-alice",
			"key":             "sk-old-key-0000000000",
			"description":     "attempted new description",
			"is_active":       false,
		},
	)
	d.SetId("map-abc-123")

	err := resourceLiteLLMJWTKeyMappingUpdate(d, client)
	if err == nil {
		t.Fatal("expected the update failure to surface as an error")
	}
	if !strings.Contains(err.Error(), "failed to refresh state afterward") {
		t.Fatalf("expected the error to mention the failed recovery read, got %v", err)
	}

	if d.Get("description").(string) != "old description" {
		t.Fatalf("a rejected description must not survive when the recovery read also fails, got %q", d.Get("description").(string))
	}
	if d.Get("is_active").(bool) != true {
		t.Fatalf("a rejected is_active must not survive when the recovery read also fails, got %v", d.Get("is_active").(bool))
	}
}

func TestJWTKeyMappingReadPopulatesStateAndKeepsKey(t *testing.T) {
	srv, calls := jwtKeyMappingTestServer(t, jwtKeyMappingFixture())
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-configured-value",
	})
	d.SetId("map-abc-123")

	if err := resourceLiteLLMJWTKeyMappingRead(d, client); err != nil {
		t.Fatalf("read failed: %v", err)
	}

	read := (*calls)[0]
	if read.Method != "GET" || read.Path != "/jwt/key/mapping/info" {
		t.Fatalf("expected GET /jwt/key/mapping/info, got %s %s", read.Method, read.Path)
	}
	if read.Query != "id=map-abc-123" {
		t.Fatalf("expected the mapping id in the query, got %q", read.Query)
	}

	if d.Get("jwt_claim_value").(string) != "dev-alice" {
		t.Fatalf("claim value not populated: %q", d.Get("jwt_claim_value").(string))
	}
	if d.Get("description").(string) != "dev-alice" {
		t.Fatalf("description not populated: %q", d.Get("description").(string))
	}
	if !d.Get("is_active").(bool) {
		t.Fatal("is_active not populated")
	}
	if d.Get("created_at").(string) != "2026-08-06T10:00:00Z" || d.Get("created_by").(string) != "admin" {
		t.Fatalf("computed audit fields not populated: %v", d.State().Attributes)
	}
	if d.Get("key").(string) != "sk-configured-value" {
		t.Fatalf("the API never returns the key, so the configured value must survive a read, got %q", d.Get("key").(string))
	}
}

func TestJWTKeyMappingReadClearsIDWhenMappingIsGone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]string{"detail": "Mapping not found"})
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-abc123",
	})
	d.SetId("map-gone")

	if err := resourceLiteLLMJWTKeyMappingRead(d, client); err != nil {
		t.Fatalf("a deleted mapping must not fail the read: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected the id to be cleared so Terraform plans a recreate, got %q", d.Id())
	}
}

func TestJWTKeyMappingUpdateClearsDescriptionAndSendsKey(t *testing.T) {
	mapping := jwtKeyMappingFixture()
	mapping.Description = ""
	srv, calls := jwtKeyMappingTestServer(t, mapping)
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-rotated",
		"is_active":       true,
	})
	d.SetId("map-abc-123")

	if err := resourceLiteLLMJWTKeyMappingUpdate(d, client); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	update := (*calls)[0]
	if update.Method != "POST" || update.Path != "/jwt/key/mapping/update" {
		t.Fatalf("expected POST /jwt/key/mapping/update, got %s %s", update.Method, update.Path)
	}
	if update.Body["id"] != "map-abc-123" {
		t.Fatalf("update must carry the mapping id, got %v", update.Body["id"])
	}
	if update.Body["key"] != "sk-rotated" {
		t.Fatalf("rotated key not sent: %v", update.Body["key"])
	}
	description, sent := update.Body["description"]
	if !sent || description != "" {
		t.Fatalf("a dropped description must be sent as an empty string, since the proxy ignores absent fields: %v", update.Body)
	}
	if d.Get("description").(string) != "" {
		t.Fatalf("description should be cleared in state, got %q", d.Get("description").(string))
	}
}

func TestJWTKeyMappingUpdateRevertsKeyOnFailureAndResyncsRest(t *testing.T) {
	// Regression test for a live-verified bug: Terraform's classic SDKv2 CRUD
	// model persists ResourceData's diff-applied (attempted) values to state
	// even when the callback returns an error, unless the provider reverts
	// them explicitly. Confirmed live: a rejected key rotation left the new,
	// never-applied key in `terraform state pull` while the proxy kept the
	// old one, so the next plan falsely reported convergence.
	calls := make([]jwtKeyMappingCall, 0)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := map[string]interface{}{}
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		calls = append(calls, jwtKeyMappingCall{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Body: body})

		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/jwt/key/mapping/update":
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"detail": "The provided key does not match an existing virtual key.",
			})
		case "/jwt/key/mapping/info":
			// Server truth: unchanged, since the rejected update above never applied.
			_ = json.NewEncoder(w).Encode(jwtKeyMappingFixture())
		default:
			t.Fatalf("unexpected request to %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)

	d := resourceDataWithChange(t,
		map[string]string{
			"id":              "map-abc-123",
			"jwt_claim_name":  "client_id",
			"jwt_claim_value": "dev-alice",
			"key":             "sk-old-key-0000000000",
			"description":     "dev-alice",
			"is_active":       "true",
		},
		map[string]interface{}{
			"jwt_claim_name":  "client_id",
			"jwt_claim_value": "dev-alice",
			"key":             "sk-rejected-new-key-00",
			"description":     "attempted new description",
			"is_active":       false,
		},
	)
	d.SetId("map-abc-123")

	err := resourceLiteLLMJWTKeyMappingUpdate(d, client)
	if err == nil {
		t.Fatal("expected the rejected key to fail the update")
	}
	if !strings.Contains(err.Error(), "does not match an existing virtual key") {
		t.Fatalf("expected the proxy's rejection reason in the error, got %v", err)
	}

	if d.Get("key").(string) != "sk-old-key-0000000000" {
		t.Fatalf("a failed update must not persist the rejected key into state, got %q", d.Get("key").(string))
	}
	if d.Get("description").(string) != "dev-alice" {
		t.Fatalf("a failed update must resync description from the server, got %q", d.Get("description").(string))
	}
	if d.Get("is_active").(bool) != true {
		t.Fatalf("a failed update must resync is_active from the server, got %v", d.Get("is_active").(bool))
	}

	readCalls := 0
	for _, c := range calls {
		if c.Path == "/jwt/key/mapping/info" {
			readCalls++
		}
	}
	if readCalls != 1 {
		t.Fatalf("expected exactly one read to resync state after the failed update, got %d", readCalls)
	}
}

func TestJWTKeyMappingUpdateOmitsMissingKeyRatherThanBlankingIt(t *testing.T) {
	srv, calls := jwtKeyMappingTestServer(t, jwtKeyMappingFixture())
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"description":     "dev-alice",
		"is_active":       true,
	})
	d.SetId("map-abc-123")

	if err := resourceLiteLLMJWTKeyMappingUpdate(d, client); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	if _, sent := (*calls)[0].Body["key"]; sent {
		t.Fatalf("a missing key must be omitted rather than blanking the mapping token: %v", (*calls)[0].Body)
	}
}

func TestJWTKeyMappingDeleteToleratesMissingMapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]string{"detail": "Mapping not found"})
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-abc123",
	})
	d.SetId("map-already-gone")

	if err := resourceLiteLLMJWTKeyMappingDelete(d, client); err != nil {
		t.Fatalf("deleting an already deleted mapping must succeed: %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("expected the id to be cleared after delete, got %q", d.Id())
	}
}

func TestJWTKeyMappingCreateSurfacesDuplicateClaimError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"detail": "A mapping for claim 'client_id' = 'dev-alice' already exists.",
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-abc123",
		"is_active":       true,
	})

	err := resourceLiteLLMJWTKeyMappingCreate(d, client)
	if err == nil {
		t.Fatal("expected a duplicate claim pair to fail")
	}
	if !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("the proxy explanation must reach the user, got %v", err)
	}
	if d.Id() != "" {
		t.Fatalf("no id should be recorded for a failed create, got %q", d.Id())
	}
}

func TestJWTKeyMappingCreateDoesNotLeakKeyInErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"key":    "sk-super-secret",
			"detail": "The provided key does not match an existing virtual key.",
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	d := schema.TestResourceDataRaw(t, resourceLiteLLMJWTKeyMapping().Schema, map[string]interface{}{
		"jwt_claim_name":  "client_id",
		"jwt_claim_value": "dev-alice",
		"key":             "sk-super-secret",
		"is_active":       true,
	})

	err := resourceLiteLLMJWTKeyMappingCreate(d, client)
	if err == nil {
		t.Fatal("expected an unknown virtual key to fail")
	}
	if !strings.Contains(err.Error(), "does not match an existing virtual key") {
		t.Fatalf("the proxy explanation must reach the user, got %v", err)
	}
	if strings.Contains(err.Error(), "sk-super-secret") {
		t.Fatalf("the virtual key must be redacted in errors, got %v", err)
	}
}
