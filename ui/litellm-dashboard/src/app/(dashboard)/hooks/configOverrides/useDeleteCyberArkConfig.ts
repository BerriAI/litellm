import { deleteCyberArkConfig } from "./cyberArkApi";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { cyberArkKeys } from "./useCyberArkConfig";

export const useDeleteCyberArkConfig = (accessToken: string | null) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return deleteCyberArkConfig(accessToken);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: cyberArkKeys.all });
    },
  });
};
