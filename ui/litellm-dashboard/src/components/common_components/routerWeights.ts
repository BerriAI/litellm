export type GroupWeights = Readonly<Record<string, number>>;
export type RouterWeights = Readonly<Record<string, GroupWeights>>;

export interface WeightableDeployment {
  model_name?: string | null;
  litellm_params?: { model?: string | null } | null;
  model_info?: { id?: string | null; team_id?: string | null; team_public_model_name?: string | null } | null;
}

export interface WeightRow {
  deploymentId: string;
  label: string | null;
  weight: number | null;
  sharePercent: number | null;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isWeight = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;

export const routerWeightsFrom = (stored: unknown): RouterWeights =>
  isRecord(stored)
    ? Object.fromEntries(
        Object.entries(stored)
          .filter((entry): entry is [string, Record<string, unknown>] => isRecord(entry[1]))
          .map(([group, weights]) => [
            group,
            Object.fromEntries(
              Object.entries(weights).filter((entry): entry is [string, number] => isWeight(entry[1])),
            ),
          ]),
      )
    : {};

/** Mirrors the proxy's canonical group check: team deployments are named by their public model name. */
const deploymentGroup = (deployment: WeightableDeployment): string | null =>
  deployment.model_info?.team_public_model_name || deployment.model_name || null;

/** Mirrors the proxy's ownership check: another team's deployment is never weightable. */
const isWeightable = (deployment: WeightableDeployment, group: string, teamId: string | null): boolean => {
  const ownerTeamId = deployment.model_info?.team_id ?? null;
  const visibleToCaller = ownerTeamId === null || ownerTeamId === teamId;
  return deploymentGroup(deployment) === group && Boolean(deployment.model_info?.id) && visibleToCaller;
};

export const withGroup = (weights: RouterWeights, group: string): RouterWeights =>
  group in weights ? weights : { ...weights, [group]: {} };

export const withoutGroup = (weights: RouterWeights, group: string): RouterWeights =>
  Object.fromEntries(Object.entries(weights).filter(([name]) => name !== group));

export const withDeploymentWeight = (
  weights: RouterWeights,
  group: string,
  deploymentId: string,
  weight: number | null,
): RouterWeights => {
  const others = Object.fromEntries(Object.entries(weights[group] ?? {}).filter(([id]) => id !== deploymentId));
  return { ...weights, [group]: weight === null ? others : { ...others, [deploymentId]: weight } };
};

export const savedRouterWeights = (weights: RouterWeights): RouterWeights | null => {
  const nonEmpty = Object.entries(weights).filter(([, group]) => Object.keys(group).length > 0);
  return nonEmpty.length > 0 ? Object.fromEntries(nonEmpty) : null;
};

export const hasNoPositiveWeight = (group: GroupWeights): boolean =>
  Object.keys(group).length > 0 && !Object.values(group).some((weight) => weight > 0);

/**
 * One row per deployment of the group, then one per weighted id the group no longer has, so a stale id
 * can still be cleared. A deployment without a weight gets none of the group's traffic once any weight is set.
 */
export const weightRows = (
  group: string,
  groupWeights: GroupWeights,
  deployments: readonly WeightableDeployment[],
  teamId: string | null,
): WeightRow[] => {
  const total = Object.values(groupWeights).reduce((sum, weight) => sum + weight, 0);
  const row = (deploymentId: string, label: string | null): WeightRow => {
    const weight = groupWeights[deploymentId] ?? null;
    return {
      deploymentId,
      label,
      weight,
      sharePercent: total > 0 ? Math.round(((weight ?? 0) / total) * 100) : null,
    };
  };
  const current = deployments
    .filter((deployment) => isWeightable(deployment, group, teamId))
    .map((deployment) => row(deployment.model_info!.id!, deployment.litellm_params?.model ?? null));
  const currentIds = new Set(current.map(({ deploymentId }) => deploymentId));
  const stale = Object.keys(groupWeights)
    .filter((deploymentId) => !currentIds.has(deploymentId))
    .map((deploymentId) => row(deploymentId, null));
  return [...current, ...stale];
};

export const parseWeightInput = (raw: string): number | null | undefined => {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  const weight = Number(trimmed);
  return isWeight(weight) ? weight : undefined;
};
