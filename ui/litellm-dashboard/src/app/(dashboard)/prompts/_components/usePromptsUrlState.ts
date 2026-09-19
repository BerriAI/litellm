import { parseAsInteger, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";

export const PROMPT_ENV_FILTERS = ["all", "development", "staging", "production"] as const;
export type PromptEnvFilter = (typeof PROMPT_ENV_FILTERS)[number];

const envFilterParsers = {
  env: parseAsStringLiteral(PROMPT_ENV_FILTERS).withDefault("all"),
  page: parseAsInteger,
};

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
  const [{ env }, setEnvFilter] = useQueryStates(envFilterParsers);
  const selectEnvFilter = useCallback(
    (filter: PromptEnvFilter) => void setEnvFilter({ env: filter, page: null }),
    [setEnvFilter],
  );
  return [env, selectEnvFilter];
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
      dropEditorRequest: () => void setPanel({ view: null }, { history: "replace" }),
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
      clearVersion: () => void setPromptInfo({ version: null }),
      clearEnvironment: () => void setPromptInfo({ prompt_env: null }),
    }),
    [prompt_env, version, setPromptInfo],
  );
}
