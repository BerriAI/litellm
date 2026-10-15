import { useUISettings } from "./useUISettings";

export const APPLY_USER_BUDGET_TO_TEAM_KEYS_SETTING_KEY = "apply_user_budget_to_team_keys";

export const useApplyUserBudgetToTeamKeys = (): boolean =>
  useUISettings().data?.values?.[APPLY_USER_BUDGET_TO_TEAM_KEYS_SETTING_KEY] === true;
