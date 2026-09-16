import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect, useRef, useState } from "react";

import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";

const ENDPOINT_TYPES = Object.values(EndpointType);
const NO_ENDPOINT = "none" as const;
const MODEL_STORAGE_KEY = "selectedModel";
const ENDPOINT_STORAGE_KEY = "endpointType";

type UrlEndpoint = EndpointType | typeof NO_ENDPOINT;

const URL_ENDPOINTS: readonly UrlEndpoint[] = [...ENDPOINT_TYPES, NO_ENDPOINT];

const chatUrlParsers = {
  chat_model: parseAsString,
  chat_endpoint: parseAsString,
};

type ChatUrlValues = {
  chat_model: string | null;
  chat_endpoint: string | null;
};

export interface ChatSelection {
  model: string | null;
  endpoint: EndpointType | null;
}

export type ChatSelectionUpdate = Partial<ChatSelection>;

export const toEndpointType = (value: string | null): EndpointType | null =>
  ENDPOINT_TYPES.find((endpoint) => endpoint === value) ?? null;

const toUrlEndpoint = (value: string | null): UrlEndpoint | null =>
  URL_ENDPOINTS.find((endpoint) => endpoint === value) ?? null;

const isUrlEmpty = (values: ChatUrlValues): boolean =>
  values.chat_model === null && toUrlEndpoint(values.chat_endpoint) === null;

const fromUrl = (values: ChatUrlValues): ChatSelection => {
  const model = values.chat_model || null;
  const endpoint = toUrlEndpoint(values.chat_endpoint);
  if (endpoint === NO_ENDPOINT) return { model, endpoint: null };
  return { model, endpoint: endpoint ?? EndpointType.CHAT };
};

const toUrlEndpointValue = (endpoint: EndpointType | null): UrlEndpoint | null => {
  if (endpoint === null) return NO_ENDPOINT;
  return endpoint === EndpointType.CHAT ? null : endpoint;
};

const toUrlValues = ({ model, endpoint }: ChatSelection): ChatUrlValues => ({
  chat_model: model || null,
  chat_endpoint: toUrlEndpointValue(endpoint),
});

const readStoredSelection = (): ChatSelection => ({
  model: sessionStorage.getItem(MODEL_STORAGE_KEY) || null,
  endpoint: toEndpointType(sessionStorage.getItem(ENDPOINT_STORAGE_KEY)) ?? EndpointType.CHAT,
});

const activeStoredSelection = (stored: ChatSelection | null, url: ChatUrlValues): ChatSelection | null =>
  stored !== null && isUrlEmpty(url) ? stored : null;

const isUnlisted = (model: string | null, models: readonly ModelGroup[]): boolean =>
  Boolean(model) && !models.some((option) => option.model_group === model);

interface ChatUrlStateOptions {
  simplified: boolean;
  fixedModel?: string;
}

export function useChatUrlState({ simplified, fixedModel }: ChatUrlStateOptions) {
  const [urlValues, setUrlValues] = useQueryStates(chatUrlParsers);
  const [storedSelection, setStoredSelection] = useState<ChatSelection | null>(() =>
    simplified || !isUrlEmpty(urlValues) ? null : readStoredSelection(),
  );
  const storedSelectionRef = useRef(storedSelection);

  const selection = simplified
    ? fixedSelection(fixedModel)
    : activeStoredSelection(storedSelection, urlValues) ?? fromUrl(urlValues);

  const replaceStoredSelection = useCallback((next: ChatSelection | null) => {
    storedSelectionRef.current = next;
    setStoredSelection(next);
  }, []);

  const setSelection = useCallback(
    (update: ChatSelectionUpdate) => {
      if (simplified) return;
      void setUrlValues((previous) => {
        const current = activeStoredSelection(storedSelectionRef.current, previous) ?? fromUrl(previous);
        const next = { ...current, ...update };
        if (next.model === current.model && next.endpoint === current.endpoint) return {};
        replaceStoredSelection(null);
        return toUrlValues(next);
      });
    },
    [simplified, setUrlValues, replaceStoredSelection],
  );

  const dropUnlistedModel = useCallback(
    (models: readonly ModelGroup[]) => {
      if (simplified) return;
      void setUrlValues((previous) => {
        const stored = activeStoredSelection(storedSelectionRef.current, previous);
        if (stored !== null) {
          if (isUnlisted(stored.model, models)) replaceStoredSelection({ ...stored, model: null });
          return {};
        }
        return isUnlisted(previous.chat_model, models) ? { chat_model: null } : {};
      });
    },
    [simplified, setUrlValues, replaceStoredSelection],
  );

  const hasInvalidEndpoint = urlValues.chat_endpoint !== null && toUrlEndpoint(urlValues.chat_endpoint) === null;
  useEffect(() => {
    if (!simplified && hasInvalidEndpoint) void setUrlValues({ chat_endpoint: null });
  }, [simplified, hasInvalidEndpoint, setUrlValues]);

  useEffect(() => {
    if (simplified) return;
    if (selection.endpoint === null) sessionStorage.removeItem(ENDPOINT_STORAGE_KEY);
    else sessionStorage.setItem(ENDPOINT_STORAGE_KEY, selection.endpoint);
    if (selection.model) sessionStorage.setItem(MODEL_STORAGE_KEY, selection.model);
    else sessionStorage.removeItem(MODEL_STORAGE_KEY);
  }, [simplified, selection.model, selection.endpoint]);

  return { selectedModel: selection.model, endpointType: selection.endpoint, setSelection, dropUnlistedModel };
}

function fixedSelection(fixedModel: string | undefined): ChatSelection {
  return { model: fixedModel ?? null, endpoint: EndpointType.CHAT };
}
