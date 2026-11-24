export interface ModelGroupInfo {
  readonly modelGroup: string;
  readonly providers: readonly string[];
  readonly mode: string | undefined;
  readonly maxInputTokens: number | undefined;
  readonly maxOutputTokens: number | undefined;
  readonly inputCostPerToken: number | undefined;
  readonly outputCostPerToken: number | undefined;
  readonly supportsVision: boolean;
  readonly supportsFunctionCalling: boolean;
  readonly supportedReasoningEfforts: readonly string[];
}

export type ConfigurationValues = { readonly [key: string]: unknown };

export interface ConfigurationSchemaProperty {
  readonly type: "string";
  readonly title: string;
  readonly enum: readonly string[];
  readonly enumItemLabels: readonly string[];
  readonly default: string;
  readonly group: "navigation";
}

export interface ConfigurationSchema {
  readonly properties: { readonly [key: string]: ConfigurationSchemaProperty };
}

export interface ModelDescriptor {
  readonly id: string;
  readonly name: string;
  readonly family: string;
  readonly version: string;
  readonly detail: string;
  readonly tooltip: string;
  readonly maxInputTokens: number;
  readonly maxOutputTokens: number;
  readonly imageInput: boolean;
  readonly toolCalling: boolean;
  readonly configurationSchema: ConfigurationSchema | undefined;
}

export type ModelGroupsParseResult =
  | { readonly kind: "ok"; readonly groups: readonly ModelGroupInfo[] }
  | { readonly kind: "invalid"; readonly reason: string };

export const REASONING_EFFORT_KEY = "reasoningEffort";
export const GATEWAY_DEFAULT_EFFORT = "default";
export const ASSUMED_MAX_INPUT_TOKENS = 128000;
export const ASSUMED_MAX_OUTPUT_TOKENS = 4096;
export const MARKDOWN_LINE_BREAK = "  \n";

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

const optionalNumber = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

const optionalString = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);

const stringList = (value: unknown): readonly string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

const parseGroup = (value: unknown): ModelGroupInfo | undefined => {
  if (!isRecord(value) || typeof value.model_group !== "string") {
    return undefined;
  }
  return {
    modelGroup: value.model_group,
    providers: stringList(value.providers),
    mode: optionalString(value.mode),
    maxInputTokens: optionalNumber(value.max_input_tokens),
    maxOutputTokens: optionalNumber(value.max_output_tokens),
    inputCostPerToken: optionalNumber(value.input_cost_per_token),
    outputCostPerToken: optionalNumber(value.output_cost_per_token),
    supportsVision: value.supports_vision === true,
    supportsFunctionCalling: value.supports_function_calling === true,
    supportedReasoningEfforts: stringList(value.supported_reasoning_efforts),
  };
};

export const parseModelGroups = (body: unknown): ModelGroupsParseResult => {
  if (!isRecord(body) || !Array.isArray(body.data)) {
    return { kind: "invalid", reason: "response has no data array" };
  }
  const groups = body.data.map(parseGroup).filter((group): group is ModelGroupInfo => group !== undefined);
  return { kind: "ok", groups };
};

const isChatGroup = (group: ModelGroupInfo): boolean => group.mode === undefined || group.mode === "chat";

export const formatUsdPerMillionTokens = (costPerToken: number): string => {
  const perMillion = costPerToken * 1_000_000;
  const digits = perMillion === 0 || perMillion >= 0.01 ? perMillion.toFixed(2) : perMillion.toPrecision(2);
  return `$${digits}`;
};

const priceLine = (label: string, costPerToken: number | undefined): string =>
  costPerToken === undefined ? `${label}: no price configured` : `${label}: ${formatUsdPerMillionTokens(costPerToken)} per 1M tokens`;

const pricingDetail = (group: ModelGroupInfo): string => {
  if (group.inputCostPerToken === undefined && group.outputCostPerToken === undefined) {
    return "No pricing configured";
  }
  const input = group.inputCostPerToken === undefined ? "n/a" : formatUsdPerMillionTokens(group.inputCostPerToken);
  const output = group.outputCostPerToken === undefined ? "n/a" : formatUsdPerMillionTokens(group.outputCostPerToken);
  return `${input} in / ${output} out per 1M tokens`;
};

const capitalize = (value: string): string => value.charAt(0).toUpperCase() + value.slice(1);

const effortSchema = (efforts: readonly string[]): ConfigurationSchema | undefined => {
  if (efforts.length === 0) {
    return undefined;
  }
  return {
    properties: {
      [REASONING_EFFORT_KEY]: {
        type: "string",
        title: "Reasoning Effort",
        enum: [GATEWAY_DEFAULT_EFFORT, ...efforts],
        enumItemLabels: ["Gateway default", ...efforts.map(capitalize)],
        default: GATEWAY_DEFAULT_EFFORT,
        group: "navigation",
      },
    },
  };
};

const tooltipFor = (group: ModelGroupInfo): string => {
  const providers = group.providers.length === 0 ? "" : ` via ${group.providers.join(", ")}`;
  const context =
    group.maxInputTokens === undefined || group.maxOutputTokens === undefined
      ? `Context: unknown, assuming ${ASSUMED_MAX_INPUT_TOKENS} in / ${ASSUMED_MAX_OUTPUT_TOKENS} out tokens`
      : `Context: ${group.maxInputTokens} in / ${group.maxOutputTokens} out tokens`;
  const efforts =
    group.supportedReasoningEfforts.length === 0
      ? "Reasoning effort: not configurable"
      : `Reasoning effort: ${group.supportedReasoningEfforts.join(", ")}`;
  return [
    `LiteLLM model group ${group.modelGroup}${providers}`,
    priceLine("Input", group.inputCostPerToken),
    priceLine("Output", group.outputCostPerToken),
    context,
    efforts,
  ].join(MARKDOWN_LINE_BREAK);
};

const describeGroup = (group: ModelGroupInfo): ModelDescriptor => ({
  id: group.modelGroup,
  name: group.modelGroup,
  family: group.modelGroup,
  version: "1.0",
  detail: pricingDetail(group),
  tooltip: tooltipFor(group),
  maxInputTokens: group.maxInputTokens ?? ASSUMED_MAX_INPUT_TOKENS,
  maxOutputTokens: group.maxOutputTokens ?? ASSUMED_MAX_OUTPUT_TOKENS,
  imageInput: group.supportsVision,
  toolCalling: group.supportsFunctionCalling,
  configurationSchema: effortSchema(group.supportedReasoningEfforts),
});

export const describeModels = (groups: readonly ModelGroupInfo[]): readonly ModelDescriptor[] =>
  groups.filter(isChatGroup).map(describeGroup);

export const reasoningEffortFrom = (configuration: ConfigurationValues | undefined): string | undefined => {
  const effort = configuration?.[REASONING_EFFORT_KEY];
  return typeof effort === "string" && effort !== GATEWAY_DEFAULT_EFFORT ? effort : undefined;
};

export const estimateTokens = (text: string): number => Math.ceil(text.length / 4);
