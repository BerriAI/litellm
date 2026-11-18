/**
 * Helpers for reading batch-specific fields off a spend log row.
 *
 * The proxy's batch cost poller (CheckBatchCost) writes one spend log per completed batch
 * with request_id "<batch_id>_batch_cost" and call_type "aretrieve_batch", carrying
 * batch_models / batch_successful_requests / batch_failed_requests in metadata
 * (see litellm/proxy/spend_tracking/spend_tracking_utils.py).
 */

import { BATCH_CALL_TYPES } from "./constants";

export const BATCH_COST_REQUEST_ID_SUFFIX = "_batch_cost";

export interface BatchRequestCounts {
  successful: number;
  failed: number;
}

export const isBatchCallType = (callType: string): boolean => BATCH_CALL_TYPES.includes(callType);

const readMetaNumber = (metadata: Record<string, unknown> | undefined, key: string): number | undefined => {
  const value = metadata?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
};

/**
 * Per-request outcome counts of a batch cost row. Undefined when the row carries neither
 * count (a non-batch row, or a batch logged before counts were tracked).
 */
export const getBatchRequestCounts = (
  metadata: Record<string, unknown> | undefined,
): BatchRequestCounts | undefined => {
  const successful = readMetaNumber(metadata, "batch_successful_requests");
  const failed = readMetaNumber(metadata, "batch_failed_requests");
  if (successful === undefined && failed === undefined) return undefined;
  return { successful: successful ?? 0, failed: failed ?? 0 };
};

/** The provider batch id behind a poller-written "<batch_id>_batch_cost" spend row. */
export const getBatchIdFromRequestId = (requestId: string): string | undefined =>
  requestId.endsWith(BATCH_COST_REQUEST_ID_SUFFIX) && requestId.length > BATCH_COST_REQUEST_ID_SUFFIX.length
    ? requestId.slice(0, -BATCH_COST_REQUEST_ID_SUFFIX.length)
    : undefined;

/** The models the batch's requests actually ran on, from metadata.batch_models. */
export const getBatchModels = (metadata: Record<string, unknown> | undefined): string[] | undefined => {
  const models = metadata?.["batch_models"];
  if (!Array.isArray(models)) return undefined;
  const names = models.filter((model): model is string => typeof model === "string" && model !== "");
  return names.length > 0 ? names : undefined;
};

/**
 * Reasoning tokens aggregated across the row's completion usage. Read from the same two
 * metadata containers the drawer already uses for prompt-token details: per-request rows
 * carry additional_usage_values, batch cost rows carry usage_object.
 */
export const getReasoningTokens = (metadata: Record<string, unknown> | undefined): number | undefined => {
  const readDetails = (container: unknown): number | undefined => {
    if (typeof container !== "object" || container === null) return undefined;
    const details = (container as Record<string, unknown>)["completion_tokens_details"];
    if (typeof details !== "object" || details === null) return undefined;
    const reasoning = (details as Record<string, unknown>)["reasoning_tokens"];
    return typeof reasoning === "number" && Number.isFinite(reasoning) ? reasoning : undefined;
  };
  return readDetails(metadata?.["additional_usage_values"]) ?? readDetails(metadata?.["usage_object"]);
};
