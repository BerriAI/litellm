import { useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { modelAccessGroupKeys } from "./useModelAccessGroups";

// ── Types ────────────────────────────────────────────────────────────────────

export type SetModelAccessGroupBudgetParams = components["schemas"]["AccessGroupBudgetRequest"];
type SetModelAccessGroupBudgetResponse = components["schemas"]["AccessGroupBudgetResponse"];

export interface SetModelAccessGroupBudgetVariables {
  accessGroup: string;
  params: SetModelAccessGroupBudgetParams;
}

// ── Fetch function ───────────────────────────────────────────────────────────

const setModelAccessGroupBudget = async ({
  accessGroup,
  params,
}: SetModelAccessGroupBudgetVariables): Promise<SetModelAccessGroupBudgetResponse | undefined> => {
  const { data } = await fetchClient.PUT("/access_group/{access_group}/budget", {
    params: { path: { access_group: accessGroup } },
    body: params,
  });
  return data;
};

// ── Hook ─────────────────────────────────────────────────────────────────────

/** Set or replace a model access group's shared budget. The write is idempotent. */
export const useSetModelAccessGroupBudget = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: setModelAccessGroupBudget,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelAccessGroupKeys.all });
    },
  });
};
