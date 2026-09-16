import { parseAsInteger, parseAsString, parseAsStringLiteral, useQueryState, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";

export const PROMPT_ENV_FILTERS = ["all", "development", "staging", "production"] as const;
export type PromptEnvFilter = (typeof PROMPT_ENV_FILTERS)[number];

const envFilterParser = parseAsStringLiteral(PROMPT_ENV_FILTERS).withDefault("all");

const panelParsers = {
  prompt: parseAsString,
  prompt_env: parseAsString,
  view: parseAsStringLiteral(["editor"] as const),
  version: parseAsInteger,
  tab: parseAsString,
};

const promptInfoParsers = {
  prompt_env: parseAsString,
  version: parseAsInteger,
};

const CLOSED_PROMPT = { prompt: null, prompt_env: null, view: null, version: null, tab: null } as const;

export const toPromptEnvFilter = (value: unknown): PromptEnvFilter =>
  PROMPT_ENV_FILTERS.find((filter) => filter === value) ?? "all";

export function usePromptEnvFilter(): [PromptEnvFilter, (filter: PromptEnvFilter) => void] {
  const [envFilter, setEnvFilter] = useQueryState("env", envFilterParser);
  const selectEnvFilter = useCallback((filter: PromptEnvFilter) => void setEnvFilter(filter), [setEnvFilter]);
  return [envFilter, selectEnvFilter];
}

export function usePromptsPanelUrlState() {
  const [{ prompt, prompt_env, view }, setPanel] = useQueryStates(panelParsers, { history: "push" });
  return useMemo(
    () => ({
      promptId: prompt,
      promptEnvironment: prompt_env,
      isEditorRequested: view === "editor",
      openPrompt: (promptId: string, environment: string) =>
        void setPanel({ ...CLOSED_PROMPT, prompt: promptId, prompt_env: environment }),
      closePrompt: () => void setPanel(CLOSED_PROMPT),
      openNewPromptEditor: () => void setPanel({ ...CLOSED_PROMPT, view: "editor" }),
      openEditor: () => void setPanel({ view: "editor" }),
      closeEditor: () => void setPanel({ view: null }),
    }),
    [prompt, prompt_env, view, setPanel],
  );
}

export function usePromptInfoUrlState() {
  const [{ prompt_env, version }, setPromptInfo] = useQueryStates(promptInfoParsers);
  return useMemo(
    () => ({
      environment: prompt_env || null,
      version,
      selectEnvironment: (environment: string) => void setPromptInfo({ prompt_env: environment, version: null }),
      selectVersion: (nextVersion: number | null, environment: string) =>
        void setPromptInfo({ version: nextVersion, prompt_env: environment }),
    }),
    [prompt_env, version, setPromptInfo],
  );
}
