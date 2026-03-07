import { deleteHashicorpVaultConfig } from "@/components/networking";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { hashicorpVaultKeys } from "./useHashicorpVaultConfig";

export const useDeleteHashicorpVaultConfig = (accessToken: string | null) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return deleteHashicorpVaultConfig(accessToken);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: hashicorpVaultKeys.all });
    },
  });
};
