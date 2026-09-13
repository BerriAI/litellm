# Resource: litellm_team_member_add

Add multiple members to a team with a single resource. This resource efficiently manages team members by using the appropriate API endpoints for each operation:

- **Adding new members**: Uses `/team/member_add` endpoint
- **Updating existing members**: Uses `/team/member_update` endpoint (preserves member identity)
- **Removing members**: Uses `/team/member_delete` endpoint

When you modify an existing team member's attributes (like role), the resource will update the member in-place rather than deleting and re-adding them.

## Example Usage

### Basic Usage

```hcl
resource "litellm_team_member_add" "example" {
  team_id = "team-123"
  
  member {
    user_id = "user-456"
    role    = "admin"
  }

  member {
    user_email = "user@example.com"
    role       = "user"
  }

  max_budget_in_team = 100.0
}
```

### Complete Team Setup with Members

```hcl
# First create a team
resource "litellm_team" "development" {
  team_alias = "development-team"
  max_budget = 500.0
  models     = ["gpt-4", "gpt-3.5-turbo"]
  
  team_member_permissions = [
    "create_key",
    "view_spend"
  ]
}

# Add members to the team
resource "litellm_team_member_add" "dev_team_members" {
  team_id = litellm_team.development.id
  
  # Team lead with admin role
  member {
    user_email = "team-lead@company.com"
    role       = "admin"
  }
  
  # Regular developers
  member {
    user_email = "developer1@company.com"
    role       = "user"
  }
  
  member {
    user_email = "developer2@company.com"
    role       = "user"
  }
  
  member {
    user_id = "existing-user-123"
    role    = "user"
  }
  
  # Budget per member
  max_budget_in_team = 100.0
}
```

### Dynamic Members Using Locals

```hcl
locals {
  team_members = [
    {
      user_id  = "user-123"
      role     = "admin"
    },
    {
      user_email = "developer1@company.com"
      role      = "user"
    },
    {
      user_email = "developer2@company.com"
      role      = "user"
    }
  ]
}

resource "litellm_team_member_add" "dynamic_example" {
  team_id = "team-456"
  
  dynamic "member" {
    for_each = local.team_members
    content {
      user_id    = lookup(member.value, "user_id", null)
      user_email = lookup(member.value, "user_email", null)
      role       = member.value.role
    }
  }

  max_budget_in_team = 200.0
}
```

### Budget Update Example

```hcl
# This example demonstrates how budget updates work correctly
resource "litellm_team_member_add" "budget_example" {
  team_id = litellm_team.example.id
  
  # Initial budget of $100 per member
  max_budget_in_team = 100.0
  
  member {
    user_email = "user1@example.com"
    role       = "admin"
  }
  
  member {
    user_email = "user2@example.com"
    role       = "user"
  }
  
  member {
    user_id = "user123"
    role    = "user"
  }
}

# To update the budget:
# 1. Change max_budget_in_team from 100.0 to 120.0
# 2. Run terraform plan - it will show the budget change
# 3. Run terraform apply - all existing members will be updated with the new budget
```

## Argument Reference

* `team_id` - (Required) The ID of the team to add members to.
* `member` - (Required) One or more member blocks defining team members. Each block supports:
  * `user_id` - (Optional) The ID of the user to add to the team.
  * `user_email` - (Optional) The email of the user to add to the team.
  * `role` - (Required) The role of the user in the team. Must be one of: "admin" or "user".
* `max_budget_in_team` - (Optional) The maximum budget allocated for the team members.

## Import

Team members can be imported using a composite ID of the team ID and user ID:

```shell
terraform import litellm_team_member_add.example team-123:user-456
