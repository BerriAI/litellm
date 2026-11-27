import type { RetryPolicyPayload } from "@/app/(dashboard)/hooks/routerSettings/useUpdateRetryPolicy";
import { RetryPolicyDraft } from "./schema";

export interface RouterRetrySettings extends RetryPolicyPayload {
  num_retries?: number | null;
}

export function toRetrySettings(settings: RouterRetrySettings) {
  return {
    draft: new RetryPolicyDraft({
      retry_policy: settings.retry_policy ?? null,
      model_group_retry_policy: settings.model_group_retry_policy ?? null,
    }),
    defaultRetry: settings.num_retries ?? 2,
  };
}

export function toRetryPolicyPayload(draft: RetryPolicyDraft): RetryPolicyPayload {
  return draft.policies;
}
