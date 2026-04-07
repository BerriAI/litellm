import { useMutation, useQueryClient } from "@tanstack/react-query";
import { setCallbacksCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { callbackKeys } from "./useCallbacks";

interface CallbackPayload {
  environment_variables: Record<string, string>;
  litellm_settings: Record<string, string[]>;
}

interface UpdateCallbackParams {
  payload: CallbackPayload;
}

interface UpdateCallbackResponse {
  message: string;
}

export const useUpdateCallback = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<UpdateCallbackResponse, Error, UpdateCallbackParams>({
    mutationFn: async ({ payload }) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return setCallbacksCall(accessToken, payload);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: callbackKeys.all });
    },
  });
};
