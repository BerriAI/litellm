package litellm

import (
	"fmt"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/resource"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

func TestAccLiteLLMOrganizationMemberAdd_basic(t *testing.T) {
	resource.Test(t, resource.TestCase{
		PreCheck:  func() { testAccPreCheck(t) },
		Providers: testAccProviders,
		Steps: []resource.TestStep{
			{
				Config: testAccLiteLLMOrganizationMemberAddConfig("test-org-bulk", "bulk-user-1", "bulk-user-2"),
				Check: resource.ComposeTestCheckFunc(
					testAccCheckLiteLLMOrganizationMemberAddExists("litellm_organization_member_add.test_members"),
					resource.TestCheckResourceAttr("litellm_organization_member_add.test_members", "member.#", "2"),
				),
			},
		},
	})
}

func testAccCheckLiteLLMOrganizationMemberAddExists(n string) resource.TestCheckFunc {
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

func testAccLiteLLMOrganizationMemberAddConfig(orgAlias, user1, user2 string) string {
	return fmt.Sprintf(`
resource "litellm_model" "test_model" {
  model_name          = "gpt-3.5-turbo"
  custom_llm_provider = "openai"
  base_model          = "gpt-3.5-turbo"
}

resource "litellm_organization" "test_org_bulk" {
  organization_alias = "%s"
  max_budget         = 100.0
  budget_duration    = "30d"
  
  depends_on = [litellm_model.test_model]
}

resource "litellm_organization_member_add" "test_members" {
  organization_id = litellm_organization.test_org_bulk.id
  
  member {
    user_id    = "%s"
    user_email = "%s@example.com"
    role       = "org_admin"
  }

  member {
    user_id    = "%s"
    user_email = "%s@example.com"
    role       = "internal_user"
  }
}
`, orgAlias, user1, user1, user2, user2)
}
