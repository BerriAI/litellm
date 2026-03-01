import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [Beta] Project Management UI

Manage projects directly from the LiteLLM Admin UI. Projects sit between teams and keys in your organizational hierarchy, enabling fine-grained access control and budget management for specific use cases or applications.

:::info
Project Management is a beta feature. The API and UI are subject to change. For the full API documentation, see [Project Management](./project_management.md).
:::

## Overview

Projects enable you to:

- Organize API keys by use case or application
- Set project-level budgets and rate limits
- Track spend and usage at the project level
- Control which models each project can access
- Maintain clear separation between different applications or teams

**Hierarchy**: `Organizations > Teams > Projects > Keys`

For detailed information about the project API and configuration, see [Project Management](./project_management.md).

## Prerequisites

- Admin or Team Admin access
- At least one team created (projects belong to teams)
- The LiteLLM Admin UI running locally or remote

## Enable Projects in UI Settings

Before you can create projects, you need to enable the Projects feature in the Admin UI settings.

### Step 1: Access Admin Settings

Navigate to the Admin UI (e.g., `http://localhost:4000/ui/?login=success`).

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/b8de4dbf-a23b-4979-84a3-95fe17427b5a/ascreenshot_84dcb13b57a84fd589dff2d5af58adde_text_export.jpeg)

### Step 2: Open Settings Menu

Click the **"New"** button in the top navigation.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/b8de4dbf-a23b-4979-84a3-95fe17427b5a/ascreenshot_447c8ea124f64d0eb18d3c9621f7cbbc_text_export.jpeg)

### Step 3: Navigate to Admin Settings

Click **"Admin Settings"**.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/cc2ce9d9-d2d2-49f3-9fb8-c546fb8dfdcf/ascreenshot_fd792e9dbda24e7eb5cdb508c4f181f8_text_export.jpeg)

### Step 4: Open UI Settings

Click **"UI Settings New"**.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/d667f4b4-300b-47c6-9d76-12e439519da6/ascreenshot_3f3db4df432843a48b53ae16b311e7df_text_export.jpeg)

### Step 5: Enable Projects Feature

Click the toggle to enable the Projects feature.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/4819f76b-4855-4f5c-8c4b-b4c272399724/ascreenshot_9df0555ae6db425ab839d73485ee9b99_text_export.jpeg)

Once enabled, the Projects section will appear in your Admin UI navigation, and you'll be able to create and manage projects.

## Create and Manage Projects

After enabling the Projects feature, you can create projects from the Projects page.

### Step 1: Navigate to Projects

Click **"Projects New"** in the sidebar.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/889e2e55-af7a-42f1-90d5-8bba8efaa986/ascreenshot_c42e33e2226c4e8b8e8ea83a7c8955e4_text_export.jpeg)

### Step 2: Create a New Project

Click **"Create Project"**.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/8ecb531c-8e96-443d-ba1d-1a9e04ba2da3/ascreenshot_74f1b3c1c1b84517ae51881a050df73a_text_export.jpeg)

### Step 3: Enter Project Name

Click the **"Project Name"** field and enter a name for your project.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/83bf0612-2b19-4b28-ae02-bdb122dca4fa/ascreenshot_16ca328a71f04a79bb9641ab9c1ed6fe_text_export.jpeg)

### Step 4: Select a Team

Choose which team this project belongs to. Projects are scoped to teams, so you can only access models and features available to that team.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/653c2f1e-5140-49b8-962f-a2b112f4834c/ascreenshot_7861310ad77d4859adcae789a9d51bd0_text_export.jpeg)

### Step 5: Configure Model Access

Select which models this project has access to. Available models are scoped to the team's allowed models.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/401a5716-ea16-4744-866a-d0ed6007065d/ascreenshot_a936c3ca417a49b2b603c890dee9d0ea_text_export.jpeg)

### Step 6: Create Project

Click **"Create Project"** to save your project.

![](https://colony-recorder.s3.amazonaws.com/files/2026-03-01/2f9f9ba1-df0b-4bef-b17c-77dfc38372f7/ascreenshot_933e4c1b119d43beb84161b94b17b764_text_export.jpeg)

## Use Cases

### Key Organization Within Teams

Organize API keys within a team by use case or application. Group related keys together in projects so you can manage budgets, model access, and permissions as a unit instead of individually.

### Cost Allocation

Assign projects to different cost centers or teams. Track spend per project and allocate costs back to the responsible team or business unit.

### Feature Rollout

Create a dedicated project for new features or experimental use cases. Control which models are available and set conservative rate limits during testing.

### Customer Segmentation

If you're a platform, create projects for different customer segments or use cases. Control resource allocation independently for each segment.

## Next Steps

After creating a project:

1. **Generate API Keys** – Create API keys scoped to your project for application use
2. **Set Budgets** – Configure project-level budget limits via the [Project Management API](./project_management.md)
3. **Track Spend** – View project-level spend in the Usage dashboard
4. **Manage Access** – Use [Access Groups](./access_groups.md) to control model and MCP server access

## Related Documentation

- [Project Management API](./project_management.md) – Full API reference for projects
- [Access Groups](./access_groups.md) – Define reusable access controls for models, MCP servers, and agents
- [Virtual Keys](./virtual_keys.md) – Create and manage API keys scoped to projects
- [Role-based Access Control](./access_control.md) – Organizations, teams, and user roles
- [Spend Logs](./spend_logs.md) – Track detailed request-level costs and usage
