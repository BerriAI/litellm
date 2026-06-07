import { useQueryClient } from "@tanstack/react-query";
import { $api, authHeader } from "@/lib/http/api";
import { handleError, deriveErrorMessage } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export const useDeleteAccessGroup = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  const mutation = $api.useMutation("delete", "/v1/access_group/{access_group_id}", {
    onError: (error) => handleError(deriveErrorMessage(error)),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["get", "/v1/access_group"] }),
  });

  return {
    ...mutation,
    mutate: (accessGroupId: string, options?: Parameters<typeof mutation.mutate>[1]) =>
      mutation.mutate(
        { params: { path: { access_group_id: accessGroupId } }, headers: authHeader(accessToken!) },
        options,
      ),
    mutateAsync: (accessGroupId: string, options?: Parameters<typeof mutation.mutateAsync>[1]) =>
      mutation.mutateAsync(
        { params: { path: { access_group_id: accessGroupId } }, headers: authHeader(accessToken!) },
        options,
      ),
  };
};
