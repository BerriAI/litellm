import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useQuery } from "@tanstack/react-query";
import { fetchDecisionModels } from "../../llm_calls/system_one";

export type ApiKeySource = "session" | "custom";

export interface DecisionModelsLookup {
  decisionModels: string[];
  isLoaded: boolean;
}

const TYPED_KEY_DEBOUNCE_WAIT_MS = 400;
const NO_DECISION_MODELS: string[] = [];

export function useDecisionModels(
  apiKeySource: ApiKeySource,
  apiKey: string,
  customBaseUrl: string | undefined,
): DecisionModelsLookup {
  const typedApiKey = apiKeySource === "custom" ? apiKey : "";
  const [settledTypedApiKey] = useDebouncedValue(typedApiKey, { wait: TYPED_KEY_DEBOUNCE_WAIT_MS });
  const lookupApiKey = apiKeySource === "session" ? apiKey : settledTypedApiKey;
  const queryOptions = {
    queryKey: ["playground", "systemOne", "decisionModels", customBaseUrl ?? "", lookupApiKey],
    queryFn: ({ signal }: { signal: AbortSignal }) => fetchDecisionModels(lookupApiKey, customBaseUrl, signal),
    enabled: lookupApiKey !== "",
    retry: false,
    staleTime: 60 * 1000,
  };
  const query = useQuery(queryOptions);
  return { decisionModels: query.data ?? NO_DECISION_MODELS, isLoaded: query.isSuccess };
}
