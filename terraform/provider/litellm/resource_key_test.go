package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/go-cty/cty"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func TestResourceKeyCreateUsesConfiguredKey(t *testing.T) {
	var payload Key
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/key/generate":
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Error(err)
			}
		case "/key/info":
		default:
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(Key{Key: "sk-customer-managed", TokenID: "token-id"})
	}))
	defer server.Close()

	d := resourceDataWithRawKey(t, "sk-customer-managed")
	diags := resourceKeyCreate(context.Background(), d, NewClient(server.URL, "sk-master", false))

	if diags.HasError() {
		t.Fatalf("create returned diagnostics: %v", diags)
	}
	if payload.Key != "sk-customer-managed" {
		t.Fatalf("configured key was dropped from create payload: got %q", payload.Key)
	}
}

func TestMapResourceDataToKeyPreservesUpdateTokenID(t *testing.T) {
	d := resourceDataWithRawKey(t, "sk-customer-managed")
	key := &Key{Key: "token-id"}

	mapResourceDataToKey(d, key)

	if key.Key != "token-id" {
		t.Fatalf("update token ID was overwritten: got %q", key.Key)
	}
}

func resourceDataWithRawKey(t *testing.T, key string) *schema.ResourceData {
	t.Helper()
	d, err := schema.InternalMap(resourceKey().Schema).Data(nil, &terraform.InstanceDiff{
		RawConfig: cty.ObjectVal(map[string]cty.Value{
			"key": cty.StringVal(key),
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	return d
}
