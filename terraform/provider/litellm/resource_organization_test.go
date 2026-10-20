package litellm

import (
	"fmt"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/resource"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func TestAccLiteLLMOrganization_basic(t *testing.T) {
	resource.Test(t, resource.TestCase{
		PreCheck:  func() { testAccPreCheck(t) },
		Providers: testAccProviders,
		Steps: []resource.TestStep{
			{
				Config: testAccLiteLLMOrganizationConfig("test-org", "test-org-alias"),
				Check: resource.ComposeTestCheckFunc(
					testAccCheckLiteLLMOrganizationExists("litellm_organization.test"),
					resource.TestCheckResourceAttr("litellm_organization.test", "organization_alias", "test-org-alias"),
					resource.TestCheckResourceAttr("litellm_organization.test", "max_budget", "100"),
				),
			},
		},
	})
}

func testAccCheckLiteLLMOrganizationExists(n string) resource.TestCheckFunc {
	return func(s *terraform.State) error {
		rs, ok := s.RootModule().Resources[n]
		if !ok {
			return fmt.Errorf("Not found: %s", n)
		}

		if rs.Primary.ID == "" {
			return fmt.Errorf("No ID is set")
		}

		return nil
	}
}

func testAccLiteLLMOrganizationConfig(name, alias string) string {
	return fmt.Sprintf(`
resource "litellm_model" "test_model" {
  model_name          = "gpt-3.5-turbo"
  custom_llm_provider = "openai"
  base_model          = "gpt-3.5-turbo"
}

resource "litellm_organization" "test" {
  organization_alias = "%s"
  max_budget         = 100.0
  budget_duration    = "30d"
  
  depends_on = [litellm_model.test_model]
}
`, alias)
}
