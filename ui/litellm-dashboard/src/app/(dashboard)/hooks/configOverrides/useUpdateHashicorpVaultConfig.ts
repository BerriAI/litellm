import { updateHashicorpVaultConfig } from "@/components/networking";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const hashicorpVaultKeys = createQueryKeys("hashicorpVaultConfig");

export const useUpdateHashicorpVaultConfig = (accessToken: string | null) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (config: Record<string, any>) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return updateHashicorpVaultConfig(accessToken, config);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: hashicorpVaultKeys.all });
    },
  });
};
