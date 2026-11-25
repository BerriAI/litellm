import type { GeneralSettingsFieldName } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import type { StoreRequestInSpendLogsParams } from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";

import type { LoggingSettingsFormValues } from "./schema";

const MINIMUM_COUNT = 1;

const duration = (value: string): string | undefined => (value.trim() === "" ? undefined : value);

// antd InputNumber applied `min` and `precision` to the committed value rather
// than only to the display, and it did so on blur, so the raw keystrokes have
// to survive until then or "12.34" commits as 124 instead of 12.
const count = (raw: string): number | undefined => {
  const parsed = Number(raw);
  if (raw.trim() === "" || !Number.isFinite(parsed)) {
    return undefined;
  }
  return Math.max(MINIMUM_COUNT, Math.round(parsed));
};

export const countDisplay = (raw: string): string => {
  const normalized = count(raw);
  return normalized === undefined ? "" : String(normalized);
};

export const storedValuesToForm = (
  storedValue: (name: keyof LoggingSettingsFormValues) => unknown,
): LoggingSettingsFormValues => ({
  store_prompts_in_spend_logs: storedValue("store_prompts_in_spend_logs") ?? false,
  maximum_spend_logs_retention_period: String(storedValue("maximum_spend_logs_retention_period") ?? ""),
  maximum_spend_logs_cleanup_batch_size: String(storedValue("maximum_spend_logs_cleanup_batch_size") ?? ""),
  maximum_spend_logs_cleanup_max_batches: String(storedValue("maximum_spend_logs_cleanup_max_batches") ?? ""),
  maximum_spend_logs_cleanup_run_budget: String(storedValue("maximum_spend_logs_cleanup_run_budget") ?? ""),
  maximum_spend_logs_cleanup_batch_timeout: String(storedValue("maximum_spend_logs_cleanup_batch_timeout") ?? ""),
});

export const formToPayload = (formValues: LoggingSettingsFormValues): StoreRequestInSpendLogsParams => {
  const retentionPeriod = duration(formValues.maximum_spend_logs_retention_period);
  const batchSize = count(formValues.maximum_spend_logs_cleanup_batch_size);
  const maxBatches = count(formValues.maximum_spend_logs_cleanup_max_batches);
  const runBudget = duration(formValues.maximum_spend_logs_cleanup_run_budget);
  const batchTimeout = duration(formValues.maximum_spend_logs_cleanup_batch_timeout);

  return {
    store_prompts_in_spend_logs: formValues.store_prompts_in_spend_logs as boolean,
    ...(retentionPeriod !== undefined && { maximum_spend_logs_retention_period: retentionPeriod }),
    ...(batchSize !== undefined && { maximum_spend_logs_cleanup_batch_size: batchSize }),
    ...(maxBatches !== undefined && { maximum_spend_logs_cleanup_max_batches: maxBatches }),
    ...(runBudget !== undefined && { maximum_spend_logs_cleanup_run_budget: runBudget }),
    ...(batchTimeout !== undefined && { maximum_spend_logs_cleanup_batch_timeout: batchTimeout }),
  };
};

// A blank field only needs clearing when something is actually stored for it.
// Asking the proxy to clear a field it has no value for is a 400 whenever no
// general_settings row exists at all, which is the state of every deployment
// that has never saved one, so clearing unconditionally would fail the first
// save on a new proxy and take the rest of the form down with it.
export const omittedFieldNames = (
  updateParams: StoreRequestInSpendLogsParams,
  names: readonly GeneralSettingsFieldName[],
  isStored: (name: GeneralSettingsFieldName) => boolean,
): readonly GeneralSettingsFieldName[] => names.filter((name) => !(name in updateParams) && isStored(name));
