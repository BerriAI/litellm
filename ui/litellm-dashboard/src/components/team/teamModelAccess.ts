export const ALL_PROXY_MODELS = "all-proxy-models";
export const NO_DEFAULT_MODELS = "no-default-models";

export interface TeamAccessGroupModelGrant {
  access_group_id: string;
  access_group_name: string;
  models: string[];
}

export type TeamModelBadgeKind = "all-proxy" | "no-default" | "direct" | "access-group";

export interface TeamModelBadge {
  label: string;
  kind: TeamModelBadgeKind;
  tooltip: string;
}

export function normalizeTeamModelSelection(models: string[] | undefined): string[] {
  return models && models.length > 0 ? models : [NO_DEFAULT_MODELS];
}

const describeGroups = (names: string[]): string =>
  names.length > 1 ? `access groups ${names.join(", ")}` : `access group ${names[0]}`;

export function computeTeamModelBadges(
  models: string[],
  accessGroupModels: string[],
  accessGroupDetails: TeamAccessGroupModelGrant[] | undefined,
): TeamModelBadge[] {
  const grants = accessGroupDetails ?? [];
  const groupNamesFor = (model: string): string[] =>
    grants.filter((g) => g.models.includes(model)).map((g) => g.access_group_name);
  const viaGroups = (model: string): string => {
    const names = groupNamesFor(model);
    return names.length > 0 ? describeGroups(names) : "an access group";
  };

  const allProxy = models.length === 0 || models.includes(ALL_PROXY_MODELS);
  const directModels = allProxy ? [] : models.filter((m) => m !== NO_DEFAULT_MODELS);
  const groupModels = [...new Set(grants.length > 0 ? grants.flatMap((g) => g.models) : accessGroupModels)].filter(
    (m) => !directModels.includes(m),
  );

  const allProxyBadge: TeamModelBadge = {
    label: "All proxy models",
    kind: "all-proxy",
    tooltip: models.includes(ALL_PROXY_MODELS)
      ? "Granted by the All Proxy Models entry in the team's model list"
      : "The team's model list is empty, so it can access every model on the proxy",
  };
  const noDefaultBadge: TeamModelBadge = {
    label: "No default models",
    kind: "no-default",
    tooltip: "No models are granted directly. Access comes only from access groups",
  };
  const headBadge = (): TeamModelBadge[] => {
    if (allProxy) return [allProxyBadge];
    if (models.includes(NO_DEFAULT_MODELS)) return [noDefaultBadge];
    return [];
  };

  return [
    ...headBadge(),
    ...directModels.map(
      (m): TeamModelBadge => ({
        label: m,
        kind: "direct",
        tooltip:
          groupNamesFor(m).length > 0
            ? `Granted directly in the team's model list, and also via ${viaGroups(m)}`
            : "Granted directly in the team's model list",
      }),
    ),
    ...groupModels.map(
      (m): TeamModelBadge => ({
        label: m,
        kind: "access-group",
        tooltip: `Granted via ${viaGroups(m)}`,
      }),
    ),
  ];
}
