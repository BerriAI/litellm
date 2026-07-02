import { useQueryClient } from "@tanstack/react-query";
import type { components } from "@/lib/http/schema";
import { $api, authHeader } from "@/lib/http/api";
import { handleError, deriveErrorMessage } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export type AccessGroupCreateParams = components["schemas"]["AccessGroupCreateRequest"];

export const useCreateAccessGroup = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  const mutation = $api.useMutation("post", "/v1/access_group", {
    onError: (error) => handleError(deriveErrorMessage(error)),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["get", "/v1/access_group"] }),
  });

  return {
    ...mutation,
    mutate: (body: AccessGroupCreateParams, options?: Parameters<typeof mutation.mutate>[1]) =>
      mutation.mutate({ body, headers: authHeader(accessToken!) }, options),
    mutateAsync: (body: AccessGroupCreateParams, options?: Parameters<typeof mutation.mutateAsync>[1]) =>
      mutation.mutateAsync({ body, headers: authHeader(accessToken!) }, options),
  };
};
