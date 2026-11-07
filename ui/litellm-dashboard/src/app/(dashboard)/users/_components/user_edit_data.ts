import type { UserInfoV2Response } from "@/components/networking";

/** Build the legacy-compatible shape consumed by UserEditView. */
export function toUserEditData(userData: UserInfoV2Response) {
  return {
    user_id: userData.user_id,
    user_info: {
      user_email: userData.user_email,
      user_alias: userData.user_alias,
      user_role: userData.user_role,
      models: userData.models,
      max_budget: userData.max_budget,
      tpm_limit: userData.tpm_limit,
      rpm_limit: userData.rpm_limit,
      budget_duration: userData.budget_duration,
      metadata: userData.metadata,
      // Without these the per-model budget editor mounts empty and a save
      // replaces the user's existing budgets with whatever was typed.
      model_max_budget: userData.model_max_budget,
      model_max_budget_usage: userData.model_max_budget_usage,
    },
  };
}
