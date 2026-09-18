import { PerSecondCostTier } from "@/components/model_dashboard/types";

const PER_SECOND_TIER_KEY = /^output_cost_per_second_(.+)$/;

export const perSecondCostTiers = (modelInfo: Record<string, unknown> | null | undefined): PerSecondCostTier[] =>
  Object.entries(modelInfo ?? {}).flatMap(([key, value]) => {
    const resolution = PER_SECOND_TIER_KEY.exec(key)?.[1];
    return resolution !== undefined && typeof value === "number" ? [{ resolution, cost: value }] : [];
  });

interface RawLitellmParams {
  model?: string;
  custom_llm_provider?: string;
  api_base?: string;
  output_cost_per_second?: number;
  [key: string]: unknown;
}

interface RawModelInfo {
  input_cost_per_token?: number | null;
  output_cost_per_token?: number | null;
  output_cost_per_second?: number;
  max_tokens?: number;
  max_input_tokens?: number;
  [key: string]: unknown;
}

interface RawModel {
  litellm_params?: RawLitellmParams;
  model_info?: RawModelInfo;
  [key: string]: unknown;
}

const costPerMillionTokens = (costPerToken: number | null | undefined) =>
  costPerToken == null ? costPerToken : (Number(costPerToken) * 1000000).toFixed(2);

const resolveProvider = (
  litellmModelName: string | null | undefined,
  customLlmProvider: string | null | undefined,
  getProviderFromModel: (model: string) => string,
): string => {
  if (!litellmModelName) return "-";
  if (customLlmProvider) return customLlmProvider;
  const splitModel = litellmModelName.split("/");
  return splitModel.length === 1 ? getProviderFromModel(litellmModelName) : splitModel[0];
};

const transformModel = (rawModel: RawModel, getProviderFromModel: (model: string) => string) => {
  const model: RawModel = JSON.parse(JSON.stringify(rawModel));
  const litellmParams = model.litellm_params;
  const modelInfo = model.model_info;

  return {
    ...model,
    provider: resolveProvider(litellmParams?.model, litellmParams?.custom_llm_provider, getProviderFromModel),
    input_cost: modelInfo ? costPerMillionTokens(modelInfo.input_cost_per_token) : null,
    output_cost: modelInfo ? costPerMillionTokens(modelInfo.output_cost_per_token) : null,
    output_cost_per_second: litellmParams?.output_cost_per_second ?? modelInfo?.output_cost_per_second ?? null,
    output_cost_per_second_tiers: perSecondCostTiers(modelInfo),
    litellm_model_name: litellmParams?.model,
    max_tokens: modelInfo ? modelInfo.max_tokens : "Undefined",
    max_input_tokens: modelInfo ? modelInfo.max_input_tokens : "Undefined",
    api_base: litellmParams?.api_base,
    cleanedLitellmParams: Object.fromEntries(
      Object.entries(litellmParams ?? {}).filter(([key]) => key !== "model" && key !== "api_base"),
    ),
  };
};

export const transformModelData = (
  rawModelData: { data?: RawModel[] | null } | null | undefined,
  getProviderFromModel: (model: string) => string,
) => {
  if (!rawModelData?.data) return { data: [] };

  return { data: rawModelData.data.map((rawModel) => transformModel(rawModel, getProviderFromModel)) };
};
