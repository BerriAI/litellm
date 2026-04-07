import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteCallback } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { callbackKeys } from "./useCallbacks";

interface DeleteCallbackResponse {
  message: string;
  removed_callback: string;
  remaining_callbacks: string[];
  deleted_at: string;
}

export const useDeleteCallback = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<DeleteCallbackResponse, Error, string>({
    mutationFn: async (callbackName: string) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return deleteCallback(accessToken, callbackName);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: callbackKeys.all });
    },
  });
};
