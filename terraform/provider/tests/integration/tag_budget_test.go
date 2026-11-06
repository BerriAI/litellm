package integration

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"

	"github.com/BerriAI/terraform-provider-litellm/litellm"
	"github.com/google/uuid"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/resource"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func TestAccTagModelBudgetDrift(t *testing.T) {
	name := "tf-tag-budget-" + uuid.NewString()
	config := fmt.Sprintf(`
provider "litellm" {}
resource "litellm_tag" "test" {
  name = %q
  model_max_budget = jsonencode({ "gpt-4" = { max_budget = 10, budget_duration = "1d" } })
  lifecycle { ignore_changes = [budget_id] }
}
`, name)
	resource.Test(t, resource.TestCase{
		PreCheck: func() {
			for _, key := range []string{"LITELLM_API_BASE", "LITELLM_API_KEY"} {
				if os.Getenv(key) == "" {
					t.Fatalf("%s must point at a live proxy with Postgres and an enterprise license", key)
				}
			}
		},
		ProviderFactories: map[string]func() (*schema.Provider, error){
			"litellm": func() (*schema.Provider, error) { return litellm.Provider(), nil },
		},
		Steps: []resource.TestStep{
			{Config: config},
			{
				PreConfig: func() {
					client := litellm.NewClient(os.Getenv("LITELLM_API_BASE"), os.Getenv("LITELLM_API_KEY"), false)
					response, err := litellm.MakeRequest(client, http.MethodPost, "/tag/update", map[string]interface{}{
						"name": name,
						"model_max_budget": map[string]interface{}{
							"gpt-4": map[string]interface{}{"max_budget": 77, "budget_duration": "1d"},
						},
					})
					if err != nil {
						t.Fatal(err)
					}
					defer response.Body.Close()
					body, err := io.ReadAll(response.Body)
					if err != nil || response.StatusCode != http.StatusOK {
						t.Fatalf("out-of-band update failed: %s, %v", body, err)
					}
					t.Logf("POST /tag/update without budget_id set max_budget to 77: %s", body)
				},
				Config:             config,
				PlanOnly:           true,
				ExpectNonEmptyPlan: true,
			},
		},
	})
}
