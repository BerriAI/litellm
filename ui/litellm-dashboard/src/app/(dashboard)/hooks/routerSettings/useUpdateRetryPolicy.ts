import { setCallbacksCall } from "@/components/networking";
import { useMutation } from "@tanstack/react-query";

export interface RetryPolicyPayload {
  retry_policy?: Record<string, number> | null;
  model_group_retry_policy?: Record<string, Record<string, number> | undefined> | null;
}

export const useUpdateRetryPolicy = (accessToken: string | null) =>
  useMutation({
    mutationFn: async (policy: RetryPolicyPayload) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return setCallbacksCall(accessToken, { router_settings: policy });
    },
  });
