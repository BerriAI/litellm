package litellm

import (
	"context"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func TestKeyMetadataJSONSendsTypedValuesAndKeepsServerEntries(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{"a": "1", "server_side": "x"}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	newState := applyKeyUpdate(t, client,
		map[string]string{"key_alias": "alias-1", "metadata.%": "1", "metadata.a": "1"},
		map[string]interface{}{
			"key_alias":     "alias-1",
			"metadata":      map[string]interface{}{"a": "1"},
			"metadata_json": `{"enabled": true, "limits": [1, 2]}`,
		},
	)

	want := map[string]interface{}{
		"a":           "1",
		"server_side": "x",
		"enabled":     true,
		"limits":      []interface{}{float64(1), float64(2)},
	}
	if got := proxy.updates[0]["metadata"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("payload metadata = %v, want %v", got, want)
	}
	if got := newState.Attributes["metadata.%"]; got != "1" {
		t.Errorf("metadata should hold only the string entry, got %v", newState.Attributes)
	}
	if got := newState.Attributes["metadata_json"]; got != `{"enabled": true, "limits": [1, 2]}` {
		t.Errorf("metadata_json = %q, want the configured string kept when the server matches", got)
	}
}

func TestKeyMetadataJSONRemovedEntryIsDeleted(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{"flag": true, "server_side": "x"}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()
	client := NewClient(srv.URL, "test-key", true)

	applyKeyUpdate(t, client,
		map[string]string{"key_alias": "alias-1", "metadata_json": `{"flag":true}`},
		map[string]interface{}{"key_alias": "alias-1"},
	)

	want := map[string]interface{}{"server_side": "x"}
	if got := proxy.updates[0]["metadata"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("payload metadata = %v, want %v", got, want)
	}
}

func TestKeyMetadataJSONReadShowsDrift(t *testing.T) {
	proxy := &fakeKeyProxy{metadata: map[string]interface{}{"flag": false}}
	srv := httptest.NewServer(proxy.handler())
	defer srv.Close()

	d := resourceKey().TestResourceData()
	d.SetId("hash-1")
	d.Set("metadata_json", `{"flag":true}`)
	if diags := resourceKeyRead(context.Background(), d, NewClient(srv.URL, "test-key", true)); diags.HasError() {
		t.Fatalf("read failed: %v", diags)
	}
	if got := d.Get("metadata_json"); got != `{"flag":false}` {
		t.Fatalf("metadata_json = %q, want the server value", got)
	}
}

func TestTeamMetadataJSONSentTypedAndReadBack(t *testing.T) {
	var captured map[string]interface{}
	srv := newTeamTestServer(t, &captured, `{"team_id":"team-1","team_info":{"team_id":"team-1","team_alias":"eng","metadata":{"owner":"eng","flag":true}},"keys":[],"team_memberships":[]}`)
	defer srv.Close()

	d := newTeamResourceData(t, map[string]interface{}{
		"team_alias":    "eng",
		"metadata":      map[string]interface{}{"owner": "eng"},
		"metadata_json": `{"flag": true}`,
	})
	if err := resourceLiteLLMTeamCreate(d, NewClient(srv.URL, "test-key", true)); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	want := map[string]interface{}{"owner": "eng", "flag": true}
	if got := captured["metadata"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("payload metadata = %v, want %v", got, want)
	}
	if got := d.Get("metadata").(map[string]interface{}); !reflect.DeepEqual(got, map[string]interface{}{"owner": "eng"}) {
		t.Errorf("metadata = %v, want only the string entry", got)
	}
	if got := d.Get("metadata_json"); got != `{"flag": true}` {
		t.Errorf("metadata_json = %q, want the configured string", got)
	}
}

func TestMetadataJSONConflictsAreRejected(t *testing.T) {
	cases := map[string]struct {
		config map[string]interface{}
		want   string
	}{
		"key in both": {
			config: map[string]interface{}{"key_alias": "a", "metadata": map[string]interface{}{"flag": "true"}, "metadata_json": `{"flag": true}`},
			want:   "metadata and metadata_json both set [flag]",
		},
		"reserved key": {
			config: map[string]interface{}{"key_alias": "a", "metadata_json": `{"tags": ["x"]}`},
			want:   "metadata_json sets [tags]",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := resourceKey().Diff(context.Background(), nil, terraform.NewResourceConfigRaw(tc.config), nil)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Diff error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestValidateMetadataJSON(t *testing.T) {
	for _, v := range []string{`[1]`, `"x"`, `null`, `{`} {
		if _, errs := validateMetadataJSON(v, "metadata_json"); len(errs) == 0 {
			t.Errorf("validateMetadataJSON(%s) accepted a non-object", v)
		}
	}
	if _, errs := validateMetadataJSON(`{"a": true}`, "metadata_json"); len(errs) != 0 {
		t.Errorf("validateMetadataJSON rejected an object: %v", errs)
	}
}
