export interface SemanticRouterRoute {
  name?: string;
  description?: string;
  utterances?: unknown[];
}

export interface SemanticRouterConfig {
  routes?: SemanticRouterRoute[];
}

export interface SemanticRouterValidationParams {
  defaultModel: string | undefined;
  embeddingModel: string | undefined;
  routerConfig: SemanticRouterConfig | null | undefined;
}

export const getSemanticRouterError = ({
  defaultModel,
  embeddingModel,
  routerConfig,
}: SemanticRouterValidationParams): string | null => {
  if (!defaultModel) return "Please select a Default Model";
  if (!routerConfig?.routes || routerConfig.routes.length === 0)
    return "Please configure at least one route for the auto router";
  if (!embeddingModel) return "Please select an Embedding Model";
  if (routerConfig.routes.some((route) => !route.name || !route.description || (route.utterances?.length ?? 0) === 0))
    return "Please ensure all routes have a target model, description, and at least one utterance";
  return null;
};
