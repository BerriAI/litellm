import { useQueryClient } from "@tanstack/react-query";
import type { components } from "@/lib/http/schema";
import { $api, authHeader } from "@/lib/http/api";
import { handleError, deriveErrorMessage } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export type AccessGroupUpdateParams = components["schemas"]["AccessGroupUpdateRequest"];

export interface EditAccessGroupVariables {
  accessGroupId: string;
  params: AccessGroupUpdateParams;
}

export const useEditAccessGroup = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  const mutation = $api.useMutation("put", "/v1/access_group/{access_group_id}", {
    onError: (error) => handleError(deriveErrorMessage(error)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["get", "/v1/access_group"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/v1/access_group/{access_group_id}"] });
    },
  });

  return {
    ...mutation,
    mutate: ({ accessGroupId, params }: EditAccessGroupVariables, options?: Parameters<typeof mutation.mutate>[1]) =>
      mutation.mutate(
        { params: { path: { access_group_id: accessGroupId } }, body: params, headers: authHeader(accessToken!) },
        options,
      ),
    mutateAsync: (
      { accessGroupId, params }: EditAccessGroupVariables,
      options?: Parameters<typeof mutation.mutateAsync>[1],
    ) =>
      mutation.mutateAsync(
        { params: { path: { access_group_id: accessGroupId } }, body: params, headers: authHeader(accessToken!) },
        options,
      ),
  };
};
