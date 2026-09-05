import { useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { modelKeys } from "@/app/(dashboard)/hooks/models/useModels";
import { modelAccessGroupKeys } from "./useModelAccessGroups";

export type CreateModelAccessGroupParams = components["schemas"]["NewModelGroupRequest"];
type CreateModelAccessGroupResponse = components["schemas"]["NewModelGroupResponse"];

const createModelAccessGroup = async (
  params: CreateModelAccessGroupParams,
): Promise<CreateModelAccessGroupResponse | undefined> => {
  const { data } = await fetchClient.POST("/access_group/new", { body: params });
  return data;
};

/**
 * Create a model access group by tagging every database deployment of the given model names with it.
 * The proxy refuses a name that already exists and any model that only lives in config.yaml.
 */
export const useCreateModelAccessGroup = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: createModelAccessGroup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelAccessGroupKeys.all });
      queryClient.invalidateQueries({ queryKey: modelKeys.lists() });
    },
  });
};
