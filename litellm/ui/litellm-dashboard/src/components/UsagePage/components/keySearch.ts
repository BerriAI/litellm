import type { Team } from "@/components/key_team_helpers/key_list";
import { toKeyMetadata, type KeyActivityRow, type KeySpendActivityRow } from "../dailyActivityApi";
import { formatKeyLabel } from "@/components/activity_metrics";
import type { ModelActivityData } from "../types";

export const keyActivityRowsToMetrics = (
  rows: readonly (KeyActivityRow | KeySpendActivityRow)[],
  teams: Team[],
): Record<string, ModelActivityData> =>
  Object.fromEntries(
    rows.map((row) => {
      const metadata = toKeyMetadata(row.metadata);
      const metrics: ModelActivityData = {
        label: formatKeyLabel({ metadata }, row.api_key, teams),
        key_metadata: metadata,
        total_requests: row.metrics.api_requests,
        total_successful_requests: row.metrics.successful_requests,
        total_failed_requests: row.metrics.failed_requests,
        total_cache_read_input_tokens: row.metrics.cache_read_input_tokens,
        total_cache_creation_input_tokens: row.metrics.cache_creation_input_tokens,
        total_tokens: row.metrics.total_tokens,
        prompt_tokens: row.metrics.prompt_tokens,
        completion_tokens: row.metrics.completion_tokens,
        total_spend: row.metrics.spend,
        total_response_time_ms:
          "total_response_time_ms" in row.metrics ? row.metrics.total_response_time_ms : undefined,
        total_timed_requests: "timed_requests" in row.metrics ? row.metrics.timed_requests : undefined,
        top_models: [],
        daily_data: [],
      };
      return [row.api_key, metrics];
    }),
  );
