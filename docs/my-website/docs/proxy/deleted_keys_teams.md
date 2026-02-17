import Image from '@theme/IdealImage';

# Deleted Keys & Teams Audit Logs

<Image img={require('../../img/ui_deleted_keys_table.png')} />

View deleted API keys and teams along with their spend and budget information at the time of deletion for auditing and compliance purposes.

## Overview

The Deleted Keys & Teams feature provides a comprehensive audit trail for deleted entities in your LiteLLM proxy. This feature was implemented to easily allow audits of which key or team was deleted along with the spend/budget at the time of deletion.

When a key or team is deleted, LiteLLM automatically captures:

- **Deletion timestamp** - When the entity was deleted
- **Deleted by** - Who performed the deletion action
- **Spend at deletion** - The total spend accumulated at the time of deletion
- **Original budget** - The budget that was set for the entity before deletion
- **Entity details** - Key or team identification information

This information is preserved even after deletion, allowing you to maintain accurate financial records and audit trails for compliance purposes.

## Viewing Deleted Keys

### Step 1: Navigate to API Keys Page

Navigate to the API Keys page in the LiteLLM UI:

```
http://localhost:4000/ui/?login=success&page=api-keys
```

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-17/73b97ba9-0ab5-4140-aee2-05fa90463461/ascreenshot_5e6d9f05d452405c83d7a368349d087d_text_export.jpeg)

### Step 2: Access Logs Section

Click on the "Logs" menu item in the navigation.

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-17/73b97ba9-0ab5-4140-aee2-05fa90463461/ascreenshot_8ebab354b1e542e59e1082e519927edd_text_export.jpeg)

### Step 3: View Deleted Keys

Click on "Deleted Keys" to view the table of all deleted API keys.

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-17/00668558-9326-4a6f-8e87-159d54b17a72/ascreenshot_d0e50e49e9aa43d4a22ada6f12a78b12_text_export.jpeg)

### Step 4: Review Deletion Information

The Deleted Keys table includes comprehensive information about each deleted key:

- **When** the key was deleted (timestamp)
- **Who** deleted the key (user/admin information)
- **Key identification** details

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-17/8538f7c4-634e-44c8-8d7d-fafbd6da0b02/ascreenshot_6b73f9c6a52d4e40a2368ef441cf6c8f_text_export.jpeg)

### Step 5: View Financial Information

The table also displays financial information captured at the time of deletion:

- **Spend at deletion** - Total spend accumulated when the key was deleted
- **Original budget** - The budget limit that was set for the key

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-17/f8b03850-b17c-490c-a507-c3b0b6c050ab/ascreenshot_070b139f111844bba38fbed8835b097b_text_export.jpeg)

## Viewing Deleted Teams

### Step 1: Access Deleted Teams

From the Logs section, click on "Deleted Teams" to view all deleted teams.

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-17/716ce26f-09af-4a6d-99c5-921d6b6a8555/ascreenshot_d36c16f1cf894340aa8bc20ada5922ac_text_export.jpeg)

### Step 2: Review Team Deletion Information

The Deleted Teams table provides detailed information about each deleted team:

- **When** the team was deleted (timestamp)
- **Who** deleted the team (user/admin information)
- **Team identification** details

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-17/0a3f2d3f-179a-4ad7-916e-b77a13dca01d/ascreenshot_ded5970762d54528ae656421148116c4_text_export.jpeg)

### Step 3: View Team Financial Information

Similar to deleted keys, the Deleted Teams table shows financial information:

- **Spend at deletion** - Total spend accumulated when the team was deleted
- **Original budget** - The budget limit that was set for the team

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-17/5b24871f-b57e-404d-8fbe-a4b27cb2a6a0/ascreenshot_3121fbafbd6b4abf90993ce6c03c608d_text_export.jpeg)

## Use Cases

This feature is particularly useful for:

- **Financial Auditing** - Track spend and budgets for deleted entities
- **Compliance** - Maintain records of who deleted what and when
- **Cost Analysis** - Understand spending patterns before deletion
- **Accountability** - Identify which admin or user performed deletions
- **Historical Records** - Preserve financial data even after entity deletion

## Related Features

- [Audit Logs](./multiple_admins.md) - View comprehensive audit logs for all entity changes
- [UI Logs](./ui_logs.md) - View request logs and spend tracking
