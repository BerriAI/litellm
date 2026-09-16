import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useEffect, useRef, useState } from "react";

import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";

const ENDPOINT_TYPES = Object.values(EndpointType);
const NO_ENDPOINT = "none" as const;
const MODEL_STORAGE_KEY = "selectedModel";
const ENDPOINT_STORAGE_KEY = "endpointType";

const chatUrlParsers = {
  chat_model: parseAsString,
  chat_endpoint: parseAsStringLiteral([...ENDPOINT_TYPES, NO_ENDPOINT]),
};

type ChatUrlValues = {
  chat_model: string | null;
  chat_endpoint: EndpointType | typeof NO_ENDPOINT | null;
};

export interface ChatSelection {
  model: string | null;
  endpoint: EndpointType | null;
}

export type ChatSelectionUpdate = Partial<ChatSelection>;

type ChatSelectionUpdater = ChatSelectionUpdate | ((current: ChatSelection) => ChatSelectionUpdate);

export const toEndpointType = (value: string | null): EndpointType | null =>
  ENDPOINT_TYPES.find((endpoint) => endpoint === value) ?? null;

const isUrlEmpty = (values: ChatUrlValues): boolean => values.chat_model === null && values.chat_endpoint === null;

const fromUrl = (values: ChatUrlValues): ChatSelection => {
  const model = values.chat_model || null;
  if (values.chat_endpoint === NO_ENDPOINT) return { model, endpoint: null };
  return { model, endpoint: values.chat_endpoint ?? EndpointType.CHAT };
};

const toUrlEndpoint = (endpoint: EndpointType | null): ChatUrlValues["chat_endpoint"] => {
  if (endpoint === null) return NO_ENDPOINT;
  return endpoint === EndpointType.CHAT ? null : endpoint;
};

const readStoredSelection = (): ChatSelection => ({
  model: sessionStorage.getItem(MODEL_STORAGE_KEY) || null,
  endpoint: toEndpointType(sessionStorage.getItem(ENDPOINT_STORAGE_KEY)) ?? EndpointType.CHAT,
});

export const dropUnlistedModel =
  (models: readonly ModelGroup[]) =>
  (current: ChatSelection): ChatSelectionUpdate =>
    current.model !== null && !models.some((model) => model.model_group === current.model) ? { model: null } : {};

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

  const selection = simplified ? fixedSelection(fixedModel) : resolveSelection(urlValues, storedSelection);

  const setSelection = useCallback(
    (update: ChatSelectionUpdater) => {
      if (simplified) return;
      void setUrlValues((previous) => {
        const stored = storedSelectionRef.current;
        const current = stored !== null && isUrlEmpty(previous) ? stored : fromUrl(previous);
        const patch = typeof update === "function" ? update(current) : update;
        const next = { ...current, ...patch };
        if (next.model === current.model && next.endpoint === current.endpoint) return {};
        storedSelectionRef.current = null;
        setStoredSelection(null);
        return { chat_model: next.model || null, chat_endpoint: toUrlEndpoint(next.endpoint) };
      });
    },
    [simplified, setUrlValues],
  );

  useEffect(() => {
    if (simplified) return;
    if (selection.endpoint === null) sessionStorage.removeItem(ENDPOINT_STORAGE_KEY);
    else sessionStorage.setItem(ENDPOINT_STORAGE_KEY, selection.endpoint);
    if (selection.model) sessionStorage.setItem(MODEL_STORAGE_KEY, selection.model);
    else sessionStorage.removeItem(MODEL_STORAGE_KEY);
  }, [simplified, selection.model, selection.endpoint]);

  return { selectedModel: selection.model, endpointType: selection.endpoint, setSelection };
}

function fixedSelection(fixedModel: string | undefined): ChatSelection {
  return { model: fixedModel ?? null, endpoint: EndpointType.CHAT };
}

function resolveSelection(urlValues: ChatUrlValues, storedSelection: ChatSelection | null): ChatSelection {
  if (storedSelection !== null && isUrlEmpty(urlValues)) return storedSelection;
  return fromUrl(urlValues);
}
