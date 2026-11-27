import type { RetryPolicyPayload } from "@/app/(dashboard)/hooks/routerSettings/useUpdateRetryPolicy";
import { z } from "zod";

type RetryPolicies = Required<RetryPolicyPayload>;

export class RetryPolicyDraft {
  // Keep model names and policy keys opaque to React Hook Form's object cloning and comparison.
  readonly #policies: RetryPolicies;

  constructor(policies: RetryPolicies) {
    this.#policies = policies;
  }

  get policies(): RetryPolicies {
    return this.#policies;
  }
}

export const retrySettingsSchema = z.object({
  scope: z.string().nullable(),
  draft: z.instanceof(RetryPolicyDraft),
});
