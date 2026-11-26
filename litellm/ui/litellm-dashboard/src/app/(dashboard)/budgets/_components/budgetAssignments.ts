import { z } from "zod";

import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import type { ApiClient } from "@/lib/http/client";

const patternMappingShape = {
  source: z.enum(["header", "body"]),
  field: z.string().min(1).max(256),
  pattern: z.string().min(1).max(512),
  tag: z.string().min(1).max(256),
};
export const patternMappingSchema = z.object(patternMappingShape);

export type PatternMapping = z.infer<typeof patternMappingSchema>;

export type BudgetAssignment =
  | { kind: "customer"; userId: string; existing: boolean }
  | { kind: "team"; teamId: string }
  | { kind: "pattern"; source: PatternMapping["source"]; field: string; pattern: string };

const PATTERN_FIELD = "request_pattern_tag_mappings";
export const budgetTagName = (budgetId: string) => `budget:${budgetId}`;

const configSchema = z.array(
  z.object({ field_name: z.string(), field_value: z.unknown(), editable: z.boolean().default(true) }),
);

export async function loadPatternMappings(client: ApiClient, accessToken: string) {
  const response = await client.get<unknown>("/config/list", {
    accessToken,
    query: { config_type: "general_settings" },
  });
  const setting = configSchema.parse(response).find((entry) => entry.field_name === PATTERN_FIELD);
  if (!setting) throw new Error("This proxy does not support pattern budget assignments");
  return { rules: z.array(patternMappingSchema).parse(setting.field_value ?? []), editable: setting.editable };
}

export async function savePatternMappings(client: ApiClient, accessToken: string, rules: readonly PatternMapping[]) {
  return client.post<unknown>("/config/field/update", {
    accessToken,
    body: { config_type: "general_settings", field_name: PATTERN_FIELD, field_value: rules },
  });
}

export async function assignBudget(
  client: ApiClient,
  accessToken: string,
  budget: budgetItem,
  assignment: BudgetAssignment,
): Promise<void> {
  if (assignment.kind === "customer") {
    await client.post<unknown>(assignment.existing ? "/customer/update" : "/customer/new", {
      accessToken,
      body: { user_id: assignment.userId, budget_id: budget.budget_id },
    });
    return;
  }
  if (assignment.kind === "team") {
    await client.post<unknown>("/team/update", {
      accessToken,
      body: {
        team_id: assignment.teamId,
        max_budget: budget.max_budget,
        budget_duration: budget.budget_duration ?? null,
        tpm_limit: budget.tpm_limit,
        rpm_limit: budget.rpm_limit,
        tpd_limit: budget.tpd_limit,
        soft_budget: budget.soft_budget,
      },
    });
    return;
  }
  if (budget.max_budget == null) throw new Error("Set a spend limit on this budget before assigning patterns");
  const settings = await loadPatternMappings(client, accessToken);
  if (!settings.editable) throw new Error("Pattern assignments are managed in the proxy configuration file");
  const tag = budgetTagName(budget.budget_id);
  const rule = patternMappingSchema.parse({ ...assignment, tag });

  const tags = z
    .array(z.object({ name: z.string(), litellm_budget_table: z.object({ budget_id: z.string() }).nullish() }))
    .parse(await client.get<unknown>("/tag/list", { accessToken }));
  const existingTag = tags.find((entry) => entry.name === tag);
  if (existingTag && existingTag.litellm_budget_table?.budget_id !== budget.budget_id) {
    throw new Error(`Tag ${tag} already exists with a different budget`);
  }
  if (!existingTag) {
    await client.post<unknown>("/tag/new", { accessToken, body: { name: tag, budget_id: budget.budget_id } });
  }
  if (settings.rules.some((existing) => JSON.stringify(existing) === JSON.stringify(rule))) return;
  await savePatternMappings(client, accessToken, [...settings.rules, rule]);
}
