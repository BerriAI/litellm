import { updateCyberArkConfig } from "./cyberArkApi";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { cyberArkKeys } from "./useCyberArkConfig";

export const useUpdateCyberArkConfig = (accessToken: string | null) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (config: Record<string, string>) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return updateCyberArkConfig(accessToken, config);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cyberArkKeys.all });
    },
  });
};
