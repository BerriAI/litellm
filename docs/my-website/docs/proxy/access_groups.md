import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Access Groups

Access Groups simplify how you define and manage resource access across your organization. Instead of configuring models, MCP servers, and agents separately on each key or team, you create one group that bundles the resources you want to grant, then attach that group to your keys or teams.

## Overview

**Access Groups** let you define a reusable set of allowed resources—models, MCP servers, and agents—in a single place. One group can grant access to all three resource types. Simply attach the group to a key or team, and they get access to everything defined in that group.

- **Unified resource control** – One group controls access to models, MCP servers, and agents together
- **Reusable** – Define once, attach to many keys or teams
- **Easy to maintain** – Update the group (add or remove resources) and all attached keys and teams automatically reflect the change
- **Clear visibility** – See exactly which resources each group grants and which keys/teams use it

<Image img={require('../../img/ui_access_groups.png')} />

### How It Works

**Key concept:** Define resources in a group → Attach group to key or team → Key/team gets access to all resources in the group

| Resource Type   | What the group controls                                              |
| --------------- | -------------------------------------------------------------------- |
| **Models**      | Which LLM models keys/teams can use (e.g., `gpt-4`, `claude-3-opus`) |
| **MCP Servers** | Which MCP servers are available for tool calling                     |
| **Agents**      | Which agents can be invoked                                          |

## How to Create and Use Access Groups in the UI

### 1. Navigate to Access Groups

Go to the Admin UI (e.g. `http://localhost:4000/ui` or your `PROXY_BASE_URL/ui`) and click **Access Groups** in the sidebar.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/d117fdb2-18c8-49e0-91e6-1f830d2d4b85/ascreenshot_f5822a0ddac64e3383124419d0c66298_text_export.jpeg)

### 2. Create an Access Group

Click **Create Access Group** and give your group a name.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/aefb900d-d106-4436-806c-3608ad19659f/ascreenshot_3f6fed1256604fe3b7038a0778ce3342_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/0951bb93-61bd-477e-beaf-f58810f8980b/ascreenshot_f0fb5d552fd74ff8a1080e82758fcdc2_text_export.jpeg)

### 3. Define Resources in the Group

Use the tabs to select which models, MCP servers, and agents this group grants access to:

- **Models tab** – Select the LLM models
- **MCP Servers tab** – Select MCP servers (for tool calling)
- **Agents tab** – Select agents

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/37398e8f-cd50-48c9-85e2-c77b2eeb994b/ascreenshot_440ec7906c8f4199b30ef91c903960b9_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/99d36543-8582-4bb7-a34d-3d5fe0fcf12f/ascreenshot_d9983240955c496892e1f7c38c074045_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/06fc5919-5c71-4fc3-999b-da7a4800af3f/ascreenshot_db93fdf742b249dc90a4b9d5991d6097_text_export.jpeg)

### 4. Attach the Access Group to a Key

When creating or editing a virtual key, expand **Optional Settings** and select your Access Group. The key will inherit access to all models, MCP servers, and agents defined in that group.

1. Go to **Virtual Keys** and click **+ Create New Key**
2. Expand **Optional Settings**
3. In the Access Group field, select the group you created
4. Save the key

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/cdfa76ab-bf38-4ca4-a97d-2cb50fafe50b/ascreenshot_046daecb57554c28ba553cf6c01f5450_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/84f08e9c-e9d0-42aa-8317-f385190b6d7d/ascreenshot_2d239716d30f431d9ad494baf7933d6a_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/41d7b7f9-ac58-4602-b887-c35c9b419dce/ascreenshot_8abd4fef48014dd1b88848411e6d7912_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/e37b01c0-f2d7-4133-8b2f-ccc51f6769e1/ascreenshot_f495df428ad54cac9ec43b46c3dfc1b1_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-15/3fe33cad-6b64-46c3-a66e-6e6e073c3d7a/ascreenshot_f2dcc79ae8af47dd86ade2f85165d3c1_text_export.jpeg)

### 5. Attach the Access Group to a Team

You can also attach an Access Group to a team when creating or editing the team. All keys associated with that team will then have access to the resources defined in the group.

## Use Cases

### Team-based Access

Create groups like "Engineering", "Data Science", or "Product" with the models, MCP servers, and agents each team needs. Attach the group to the team—no need to configure each resource on every key.

### Environment Separation

- **Production group** – Production models, approved MCP servers, and production agents
- **Development group** – Cost-efficient models, experimental MCP tools, and dev agents

Attach the appropriate group to keys or teams based on environment.

### Simplified Onboarding

New developers get a key with an Access Group instead of manually configuring models, MCP servers, and agents. Add them to the right team or give them a key with the correct group.

### Centralized Updates

When you add a new model or MCP server to a group, every key and team attached to that group automatically gains access. Remove a resource from the group and it’s revoked everywhere at once.

## Access Group vs. Model Access Groups

LiteLLM has two related concepts:

| Feature    | **Access Groups** (this page)                                           | **Model Access Groups**                                 |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------------------------- |
| Definition | Define in the UI; one group can include models, MCP servers, and agents | Defined in config or via API; groups are model-centric  |
| Scope      | Models + MCP servers + agents                                           | Models only                                             |
| Attach to  | Keys, teams                                                             | Keys, teams                                             |
| Use when   | You want unified control over models, MCP, and agents from the UI       | You need config-based or API-based model access control |

For config-based model access with `access_groups` in `model_info`, see [Model Access Groups](./model_access_groups.md).

## Related Documentation

- [Virtual Keys](./virtual_keys.md) – Creating and managing API keys
- [Role-based Access Controls](./access_control.md) – Organizations, teams, and user roles
- [Model Access Groups](./model_access_groups.md) – Config-based model access groups
- [MCP Control](../mcp_control.md) – MCP server setup and access control
