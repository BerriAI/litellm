package litellm

import (
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
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
