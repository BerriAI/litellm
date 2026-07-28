import { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import {
  AutoRouterKind,
  EditBlockedReason,
  autoRouterCapabilities,
  autoRouterStrategy,
} from "@/components/add_model/auto_router_strategies";
import { normalizeTierModels } from "@/components/add_model/complexity_router_tiers";

export type { AutoRouterKind };

export interface AutoRouterRow {
  id: string;
  name: string;
  kind: AutoRouterKind;
  typeLabel: string;
  /** Edit needs an API-created row AND a strategy the dashboard has a form for. */
  canEdit: boolean;
  /** Delete only needs an API-created row; removing by id never reads the config. */
  canDelete: boolean;
  editBlockedReason: EditBlockedReason | null;
  targets: string[];
  defaultModel: string | null;
  createdAt: string | null;
  deployment: AutoRouterDeployment;
}

const safeParse = (value: string): unknown => {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

const asRecord = (value: unknown): Record<string, unknown> => {
  const parsed: unknown = typeof value === "string" ? safeParse(value) : value;
  return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>)
    : {};
};

const asStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];

const dedupe = (models: string[]): string[] => Array.from(new Set(models));

export const complexityTypeLabel = (config: Record<string, unknown>): string =>
  config.classifier_type === "llm" ? "LLM Classifier" : "Heuristic";

interface Presentation {
  typeLabel: string;
  targets: string[];
}

// Adaptive and quality both declare a flat pool and have no editor here, so the row reports
// what is configured rather than interpreting it.
const configManaged = (label: string, config: Record<string, unknown>): Presentation => ({
  typeLabel: label,
  targets: asStringArray(config.available_models),
});

/** How each strategy renders itself, given its own config object. */
const PRESENTERS: Record<AutoRouterKind, (config: Record<string, unknown>) => Presentation> = {
  complexity: (config) => ({
    typeLabel: complexityTypeLabel(config),
    targets: dedupe(Object.values(asRecord(config.tiers)).flatMap(normalizeTierModels)),
  }),
  semantic: (config) => {
    const routes = dedupe(
      (Array.isArray(config.routes) ? config.routes : [])
        .map((route) => asRecord(route).name)
        .filter((name): name is string => typeof name === "string" && name.length > 0),
    );
    return { typeLabel: "Semantic", targets: routes };
  },
  adaptive: (config) => configManaged("Adaptive", config),
  quality: (config) => configManaged("Quality", config),
};

export const toAutoRouterRow = (deployment: AutoRouterDeployment, index: number): AutoRouterRow => {
  const params = deployment.litellm_params ?? {};
  const info = deployment.model_info ?? {};
  const name = deployment.model_name ?? "";
  const strategy = autoRouterStrategy(params);
  const { canEdit, canDelete, editBlockedReason } = autoRouterCapabilities(params, info);

  return {
    id: info.id ?? `${name}-${index}`,
    name,
    kind: strategy.kind,
    canEdit,
    canDelete,
    editBlockedReason,
    createdAt: info.created_at ?? null,
    defaultModel: (params[strategy.defaultModelKey] as string | null | undefined) ?? null,
    deployment,
    ...PRESENTERS[strategy.kind](asRecord(params[strategy.configKey])),
  };
};

export const toAutoRouterRows = (deployments: AutoRouterDeployment[]): AutoRouterRow[] =>
  deployments.map(toAutoRouterRow);
