"use client";

import { useQuery } from "@tanstack/react-query";
import { lensQueries } from "../../data/queries";
import { useLensApi } from "../../data/LensServices";
import type { AnalysisModelInfo } from "../../model/types";
import { useAnalysisKeyInfo } from "../../settings/worker/useAnalysisKeyInfo";

export interface AnalysisModels {
  readonly models: string[];
  readonly modelDetails: AnalysisModelInfo[];
  readonly modelsLoading: boolean;
  readonly modelsError?: string;
  /** The only model a lone worker's analysis key can use, so the setup can preselect it. */
  readonly defaultModel?: string;
}

export function useAnalysisModels(): AnalysisModels {
  const api = useLensApi();
  const models = useQuery(lensQueries.models(api));
  const modelDetails = useQuery(lensQueries.modelDetails(api));
  const list = useQuery(lensQueries.list(api));
  const activeWorkers = list.data?.workers.filter((worker) => !worker.revoked) ?? [];
  const defaultKeyId = activeWorkers.length === 1 ? activeWorkers[0].analysis_key_id : undefined;
  const keyInfo = useAnalysisKeyInfo(defaultKeyId ?? undefined);
  return {
    models: models.data?.data.map((model) => model.id) ?? [],
    modelDetails: modelDetails.data?.data ?? [],
    modelsLoading: models.isLoading,
    modelsError: models.error?.message,
    defaultModel: keyInfo.data?.models.length === 1 ? keyInfo.data.models[0] : undefined,
  };
}
