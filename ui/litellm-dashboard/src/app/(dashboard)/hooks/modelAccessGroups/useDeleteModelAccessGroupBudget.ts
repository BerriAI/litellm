import { useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { modelAccessGroupKeys } from "./useModelAccessGroups";

// ── Types ────────────────────────────────────────────────────────────────────

type DeleteModelAccessGroupBudgetResponse = components["schemas"]["DeleteAccessGroupBudgetResponse"];

// ── Fetch function ───────────────────────────────────────────────────────────

const deleteModelAccessGroupBudget = async (
  accessGroup: string,
): Promise<DeleteModelAccessGroupBudgetResponse | undefined> => {
  const { data } = await fetchClient.DELETE("/access_group/{access_group}/budget", {
    params: { path: { access_group: accessGroup } },
  });
  return data;
};

// ── Hook ─────────────────────────────────────────────────────────────────────

/**
 * Clear a model access group's shared budget. The group and its deployments are untouched,
 * and the recorded spend goes with the budget row.
 */
export const useDeleteModelAccessGroupBudget = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteModelAccessGroupBudget,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelAccessGroupKeys.all });
    },
  });
};
