package litellm

import (
	"os"
	"strings"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

var testAccProviders map[string]*schema.Provider
var testAccProvider *schema.Provider

func init() {
	testAccProvider = Provider()
	testAccProviders = map[string]*schema.Provider{
		"litellm": testAccProvider,
	}
}

func TestProvider(t *testing.T) {
	if err := Provider().InternalValidate(); err != nil {
		t.Fatalf("err: %s", err)
	}
}

func TestProvider_impl(t *testing.T) {
	var _ *schema.Provider = Provider()
}

func testAccPreCheck(t *testing.T) {
	if v := os.Getenv("LITELLM_API_BASE"); v == "" {
		t.Fatal("LITELLM_API_BASE must be set for acceptance tests")
	}
	if v := os.Getenv("LITELLM_API_KEY"); v == "" {
		t.Fatal("LITELLM_API_KEY must be set for acceptance tests")
	}

	// Create test users needed for organization member tests
	createTestUsers(t)
}

func createTestUsers(t *testing.T) {
	apiBase := os.Getenv("LITELLM_API_BASE")
	apiKey := os.Getenv("LITELLM_API_KEY")

	if apiBase == "" || apiKey == "" {
		return
	}

	client := NewClient(apiBase, apiKey, false)

	// Create test users
	users := []map[string]interface{}{
		{
			"user_id":    "test-user-1",
			"user_email": "test-user-1@example.com",
			"user_role":  "internal_user",
		},
		{
			"user_id":    "bulk-user-1",
			"user_email": "bulk-user-1@example.com",
			"user_role":  "internal_user",
		},
		{
			"user_id":    "bulk-user-2",
			"user_email": "bulk-user-2@example.com",
			"user_role":  "internal_user",
		},
	}

	for _, user := range users {
		_, err := client.sendRequest("POST", "/user/new", user)
		if err != nil {
			// Silently ignore if user already exists (400 error)
			// This is expected when running tests multiple times
			errStr := err.Error()
			if !strings.Contains(errStr, "400") && !strings.Contains(errStr, "already exists") {
				t.Logf("Warning: Could not create user %s: %v", user["user_id"], err)
			}
		}
	}
}
