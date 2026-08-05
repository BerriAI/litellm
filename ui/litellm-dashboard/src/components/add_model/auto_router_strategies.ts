/**
 * Single owner of "which auto-router strategy is this deployment, and what can we do with it".
 *
 * Two independent axes decide whether a row is writable, and both must hold:
 *   1. STRATEGY  - the dashboard only has a form for complexity and semantic routers. Adaptive
 *                  and quality store their settings under their own config keys, so opening
 *                  one in the complexity/semantic editor would write the wrong shape onto it.
 *   2. ORIGIN    - a deployment defined in config.yaml reports `db_model: false`, and the API
 *                  refuses it regardless of strategy (PATCH /model/{id}/update 404s,
 *                  POST /model/delete 400s). Only rows created through the API are writable.
 *
 * Strategy order mirrors Router._is_auto_router_deployment (router.py:7589-7594): the named
 * prefixes are matched first, and only a bare `auto_router/<name>` is the semantic router.
 */

export type AutoRouterKind = "complexity" | "adaptive" | "quality" | "semantic";

export interface AutoRouterParams {
  model?: string | null;
  complexity_router_config?: unknown;
  complexity_router_default_model?: string | null;
  auto_router_config?: unknown;
  auto_router_default_model?: string | null;
  adaptive_router_config?: unknown;
  adaptive_router_default_model?: string | null;
  quality_router_config?: unknown;
  quality_router_default_model?: string | null;
}

export interface AutoRouterStrategy {
  kind: AutoRouterKind;
  /** Type-pill label. The complexity router overrides this with its classifier. */
  label: string;
  configKey: keyof AutoRouterParams;
  defaultModelKey: keyof AutoRouterParams;
  /** Whether the dashboard has a form that understands this strategy's config shape. */
  hasEditor: boolean;
  matches: (params: AutoRouterParams) => boolean;
}

const startsWith = (params: AutoRouterParams, prefix: string): boolean => params.model?.startsWith(prefix) === true;

/** Ordered; the semantic entry matches anything left and must stay last. */
export const AUTO_ROUTER_STRATEGIES: readonly AutoRouterStrategy[] = [
  {
    kind: "complexity",
    label: "Complexity",
    configKey: "complexity_router_config",
    defaultModelKey: "complexity_router_default_model",
    hasEditor: true,
    // Also matched by config presence: rows predating the canonical model string carry the
    // config without the prefix.
    matches: (p) => startsWith(p, "auto_router/complexity_router") || p.complexity_router_config != null,
  },
  {
    kind: "adaptive",
    label: "Adaptive",
    configKey: "adaptive_router_config",
    defaultModelKey: "adaptive_router_default_model",
    hasEditor: false,
    matches: (p) => startsWith(p, "auto_router/adaptive_router"),
  },
  {
    kind: "quality",
    label: "Quality",
    configKey: "quality_router_config",
    defaultModelKey: "quality_router_default_model",
    hasEditor: false,
    matches: (p) => startsWith(p, "auto_router/quality_router"),
  },
  {
    kind: "semantic",
    label: "Semantic",
    configKey: "auto_router_config",
    defaultModelKey: "auto_router_default_model",
    hasEditor: true,
    matches: () => true,
  },
] as const;

export const autoRouterStrategy = (params: AutoRouterParams | null | undefined): AutoRouterStrategy =>
  AUTO_ROUTER_STRATEGIES.find((strategy) => strategy.matches(params ?? {}))!;

export const isComplexityRouter = (params: AutoRouterParams | null | undefined): boolean =>
  autoRouterStrategy(params).kind === "complexity";

/** Any `auto_router/*` deployment, whatever its strategy. Use for listing and filtering. */
export const isAutoRouterDeployment = (params: AutoRouterParams | null | undefined): boolean =>
  params?.model?.startsWith("auto_router/") === true ||
  params?.complexity_router_config != null ||
  params?.auto_router_config != null;

/**
 * Whether EditAutoRouterModal understands this deployment. It only speaks complexity and
 * semantic, so offering it for an adaptive or quality router lets a save write
 * `auto_router_config` onto a row that stores its settings elsewhere. Gate every edit
 * affordance on this, never on `isAutoRouterDeployment`.
 */
export const hasAutoRouterEditor = (params: AutoRouterParams | null | undefined): boolean =>
  isAutoRouterDeployment(params) && autoRouterStrategy(params).hasEditor;

export interface AutoRouterDeploymentInfo {
  db_model?: boolean | null;
}

/** Why the dashboard cannot offer an edit form, or null when it can. */
export type EditBlockedReason = "config-managed" | "no-editor";

export interface AutoRouterCapabilities {
  /** Defined in config.yaml; the API refuses both update and delete for it. */
  isConfigManaged: boolean;
  canEdit: boolean;
  /** Deleting removes a row by id and never reads its config, so strategy is irrelevant. */
  canDelete: boolean;
  editBlockedReason: EditBlockedReason | null;
}

/**
 * What could be done to this deployment by anyone with permission. Deliberately excludes the
 * caller's role: the page ANDs that in, so resource capability and actor permission stay
 * separable. Derive per capability rather than exposing one "editable" boolean, because the
 * constraints differ (edit needs an editor, delete does not) and the explanation differs again.
 */
const editBlockedReasonFor = (isConfigManaged: boolean, hasEditor: boolean): EditBlockedReason | null => {
  if (isConfigManaged) return "config-managed";
  if (!hasEditor) return "no-editor";
  return null;
};

export const autoRouterCapabilities = (
  params: AutoRouterParams | null | undefined,
  modelInfo: AutoRouterDeploymentInfo | null | undefined,
): AutoRouterCapabilities => {
  const isConfigManaged = modelInfo?.db_model !== true;
  const hasEditor = autoRouterStrategy(params).hasEditor;

  return {
    isConfigManaged,
    canEdit: !isConfigManaged && hasEditor,
    canDelete: !isConfigManaged,
    editBlockedReason: editBlockedReasonFor(isConfigManaged, hasEditor),
  };
};
