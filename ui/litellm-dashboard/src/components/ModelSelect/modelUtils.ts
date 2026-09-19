import type { ModelDeploymentSummary } from "@/app/(dashboard)/hooks/models/useModels";

export interface GroupedModels {
  wildcard: string[];
  regular: string[];
}

export const splitWildcardModels = (models: string[]): GroupedModels => {
  const wildcard: string[] = [];
  const regular: string[] = [];

  for (const model of models) {
    if (model.endsWith("/*")) {
      wildcard.push(model);
    } else {
      regular.push(model);
    }
  }

  return { wildcard, regular };
};

export interface ModelOption {
  label: string;
  value: string;
  description?: string;
  disabled?: boolean;
}

const DEPLOYMENT_ID_PREVIEW_LENGTH = 8;

export const deploymentOptionLabel = (deployment: ModelDeploymentSummary): string =>
  `${deployment.modelName} · ${deployment.litellmModel} · ${deployment.id.slice(0, DEPLOYMENT_ID_PREVIEW_LENGTH)}`;

const deploymentOption = (deployment: ModelDeploymentSummary, disabled: boolean): ModelOption => ({
  label: deploymentOptionLabel(deployment),
  value: deployment.id,
  description: `Deployment ID ${deployment.id}`,
  disabled,
});

/**
 * One option per public model name. A name backed by several deployments is followed by one
 * option per deployment so a single deployment can be granted on its own. A name that is itself
 * a deployment ID (already granted upstream, for example on the key's team) renders as that
 * deployment, and its model name is then not expanded into the deployments that were not granted.
 */
export const buildModelOptions = (
  modelNames: string[],
  deployments: ModelDeploymentSummary[],
  disabled: boolean,
  plainLabel: (modelName: string) => string = (modelName) => modelName,
): ModelOption[] => {
  const deploymentsByName = deployments.reduce(
    (byName, deployment) => byName.set(deployment.modelName, [...(byName.get(deployment.modelName) ?? []), deployment]),
    new Map<string, ModelDeploymentSummary[]>(),
  );
  const deploymentsById = new Map(deployments.map((deployment) => [deployment.id, deployment]));
  const offeredIds = new Set(modelNames.filter((modelName) => deploymentsById.has(modelName)));
  const options = modelNames.flatMap((modelName): ModelOption[] => {
    const byId = deploymentsById.get(modelName);
    if (byId) return [deploymentOption(byId, disabled)];
    const sameName = deploymentsByName.get(modelName) ?? [];
    const grantedById = sameName.some((deployment) => offeredIds.has(deployment.id));
    if (sameName.length < 2 || grantedById) return [{ label: plainLabel(modelName), value: modelName, disabled }];
    return [
      { label: plainLabel(modelName), value: modelName, description: `All ${sameName.length} deployments`, disabled },
      ...sameName.map((deployment) => deploymentOption(deployment, disabled)),
    ];
  });
  return Array.from(new Map(options.map((option) => [option.value, option])).values());
};

export const resolveSelectedModelOption = (
  value: string,
  optionsByValue: ReadonlyMap<string, ModelOption>,
  deployments: ModelDeploymentSummary[],
): ModelOption => {
  const known = optionsByValue.get(value);
  if (known) return known;
  const deployment = deployments.find((candidate) => candidate.id === value);
  if (deployment) return deploymentOption(deployment, false);
  return { label: value, value };
};
