import type { DeploymentInfoRow } from "@/components/networking";

export interface DiscoveredModelRow {
  id: string;
  upstreamId: string;
  modelName: string;
  enabled: boolean;
  alternateNames: string[];
  manual: boolean;
}

/**
 * The litellm_params.model this upstream id resolves to: provider-prefixed, but never
 * double-prefixed if the discovered/typed id already carries the provider's own prefix.
 */
export const litellmModelForUpstreamId = (litellmProvider: string, upstreamId: string): string =>
  upstreamId.startsWith(`${litellmProvider}/`) ? upstreamId : `${litellmProvider}/${upstreamId}`;

export const buildDiscoveredRows = (upstreamIds: readonly string[]): DiscoveredModelRow[] =>
  upstreamIds.map((upstreamId, index) => ({
    id: `discovered-${index}-${upstreamId}`,
    upstreamId,
    modelName: upstreamId,
    enabled: true,
    alternateNames: [],
    manual: false,
  }));

export const buildManualRow = (upstreamId: string): DiscoveredModelRow => ({
  id: `manual-${upstreamId}-${Math.random().toString(36).slice(2)}`,
  upstreamId,
  modelName: upstreamId,
  enabled: true,
  alternateNames: [],
  manual: true,
});

export interface CreationResult {
  row: DiscoveredModelRow;
  status: "created" | "skipped" | "failed";
  detail?: string;
}

export interface ModelCreationPayload {
  model_name: string;
  litellm_params: { model: string; litellm_credential_name: string };
  model_info: Record<string, never>;
  blocked: boolean;
}

/** blocked is the inverse of the review table's enabled toggle -- a disabled row is still
 * created, just paused, so a later enable is a one-click unblock instead of re-discovery. */
export const buildModelCreationPayload = (
  litellmProvider: string,
  credentialName: string,
  row: DiscoveredModelRow,
): ModelCreationPayload => ({
  model_name: row.modelName,
  litellm_params: {
    model: litellmModelForUpstreamId(litellmProvider, row.upstreamId),
    litellm_credential_name: credentialName,
  },
  model_info: {},
  blocked: !row.enabled,
});

const deploymentKey = (credentialName: string, litellmModel: string): string => `${credentialName}::${litellmModel}`;

/**
 * Rows not yet created for this credential, matched against every existing deployment by
 * (litellm_credential_name, litellm_params.model) -- the partial-failure-recovery key: re-running
 * the wizard after some rows already landed must not attempt to recreate them.
 */
export const rowsPendingCreation = (
  rows: readonly DiscoveredModelRow[],
  litellmProvider: string,
  credentialName: string,
  existingDeployments: readonly DeploymentInfoRow[],
): DiscoveredModelRow[] => {
  const existingKeys = new Set(
    existingDeployments
      .filter((deployment) => deployment.litellm_params.litellm_credential_name === credentialName)
      .map((deployment) => deploymentKey(credentialName, deployment.litellm_params.model ?? "")),
  );
  return rows.filter(
    (row) =>
      !existingKeys.has(deploymentKey(credentialName, litellmModelForUpstreamId(litellmProvider, row.upstreamId))),
  );
};

export type ModelGroupAliasValue = string | { model: string; hidden?: boolean };
export type ModelGroupAliasMap = Record<string, ModelGroupAliasValue>;

export interface AliasAddition {
  alias: string;
  targetModelGroup: string;
}

export interface AliasMergeResult {
  merged: ModelGroupAliasMap;
  collisions: readonly string[];
}

/**
 * Merges new alias entries into the existing model_group_alias map for a single
 * /config/update write. Never touches an existing key's value (object-valued {model, hidden}
 * entries survive untouched) and never overwrites a name collision -- a colliding alias is
 * reported back rather than silently dropped or silently replacing what was there.
 */
export const mergeModelGroupAliases = (
  existing: ModelGroupAliasMap,
  additions: readonly AliasAddition[],
): AliasMergeResult => {
  const collisions: string[] = [];
  const added: ModelGroupAliasMap = {};
  for (const { alias, targetModelGroup } of additions) {
    if (alias in existing || alias in added) {
      collisions.push(alias);
      continue;
    }
    added[alias] = targetModelGroup;
  }
  return { merged: { ...existing, ...added }, collisions };
};

/** Flattens the review table's per-row alternate names into the alias additions
 * mergeModelGroupAliases expects, skipping rows the operator removed/left disabled-but-empty. */
export const aliasAdditionsFromRows = (rows: readonly DiscoveredModelRow[]): AliasAddition[] =>
  rows.flatMap((row) => row.alternateNames.map((alias) => ({ alias, targetModelGroup: row.modelName })));
