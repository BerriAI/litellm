import { updateWebSearchInterceptionSettings } from "@/components/networking";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const webSearchInterceptionSettingsKeys = createQueryKeys("webSearchInterceptionSettings");

export const useUpdateWebSearchInterceptionSettings = (accessToken: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (settings: Record<string, any>) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return updateWebSearchInterceptionSettings(accessToken, settings);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: webSearchInterceptionSettingsKeys.all,
      });
    },
  });
};
