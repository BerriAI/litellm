import type { ModelMaxBudget } from "@/components/key_team_helpers/ModelMaxBudgetEditor";
import type { UserPatchRequest } from "@/components/networking";

/** The subset of `UserEditView`'s submitted values that `PATCH /management/v1/users/{id}` accepts. */
export interface UserEditFormValues {
  user_email?: string | null;
  user_alias?: string | null;
  user_role?: UserPatchRequest["user_role"];
  models?: string[];
  max_budget?: string | number | null;
  budget_duration?: string | null;
  tpm_limit?: string | number | null;
  rpm_limit?: string | number | null;
  // The editor hands back parsed JSON, or the raw text when it was left empty.
  metadata?: Record<string, unknown> | string | null;
  model_max_budget?: ModelMaxBudget;
}

const emptyToNull = (value: string | null | undefined): string | null => (value ? value : null);

const toNumberOrNull = (value: string | number | null | undefined): number | null =>
  value === "" || value === null || value === undefined ? null : Number(value);

/**
 * Build the merge-patch body for one internal user.
 *
 * The endpoint reads an omitted key as "leave alone" and an explicit null as "clear", so an emptied
 * control has to become a null rather than disappear, or the save is a silent no-op. It also refuses
 * unknown keys with a 422, which is why this picks fields out rather than forwarding the form store:
 * `user_id` and the two MCP controls are form state, not columns.
 *
 * A field the form did not render is left out entirely, since only the caller knows whether the
 * operator declined to set it or never saw it. `user_role` is the one rendered field treated that
 * way: its dropdown offers no way to clear a role, so an empty one means the user never had one.
 */
export const toUserPatch = (values: UserEditFormValues): UserPatchRequest => ({
  ...("user_email" in values && { user_email: emptyToNull(values.user_email) }),
  ...("user_alias" in values && { user_alias: emptyToNull(values.user_alias) }),
  ...("budget_duration" in values && { budget_duration: emptyToNull(values.budget_duration) }),
  ...("max_budget" in values && { max_budget: toNumberOrNull(values.max_budget) }),
  ...("tpm_limit" in values && { tpm_limit: toNumberOrNull(values.tpm_limit) }),
  ...("rpm_limit" in values && { rpm_limit: toNumberOrNull(values.rpm_limit) }),
  ...(values.user_role && { user_role: values.user_role }),
  ...("models" in values && { models: values.models ?? [] }),
  ...("metadata" in values && { metadata: typeof values.metadata === "object" ? values.metadata : null }),
  ...(values.model_max_budget !== undefined && { model_max_budget: values.model_max_budget }),
});
