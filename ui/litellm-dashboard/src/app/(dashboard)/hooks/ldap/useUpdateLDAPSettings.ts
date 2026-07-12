import { updateLDAPSettings } from "@/components/networking";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ldapKeys } from "./useLDAPSettings";

export const useUpdateLDAPSettings = (accessToken: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (settings: Record<string, unknown>) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return updateLDAPSettings(accessToken, settings);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ldapKeys.all,
      });
    },
  });
};
