import type { AnalysisModelInfo, SignalConfig } from "./types";

export const SYSTEM_ONE_MODE = "evaluation";

export const signalsConfigured = (config: SignalConfig): boolean =>
  Boolean(config.model) && (config.signals?.length ?? 0) > 0;

export const systemOneModels = (details: readonly AnalysisModelInfo[]): AnalysisModelInfo[] =>
  details
    .filter((info) => info.mode === SYSTEM_ONE_MODE)
    .toSorted((a, b) => a.model_group.localeCompare(b.model_group));
