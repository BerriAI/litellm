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

export interface TeamModelBadgeCopy {
  allProxyModels: string;
  allProxyEntryTooltip: string;
  emptyModelListTooltip: string;
  noDefaultModels: string;
  noDefaultTooltip: string;
  accessGroup: (name: string) => string;
  accessGroups: (names: string[]) => string;
  fallbackAccessGroup: string;
  directAndVia: (via: string) => string;
  direct: string;
  via: (via: string) => string;
}

const DEFAULT_BADGE_COPY: TeamModelBadgeCopy = {
  allProxyModels: "All proxy models",
  allProxyEntryTooltip: "Granted by the All Proxy Models entry in the team's model list",
  emptyModelListTooltip: "The team's model list is empty, so it can access every model on the proxy",
  noDefaultModels: "No default models",
  noDefaultTooltip: "No models are granted directly. Access comes only from access groups",
  accessGroup: (name) => `access group ${name}`,
  accessGroups: (names) => `access groups ${names.join(", ")}`,
  fallbackAccessGroup: "an access group",
  directAndVia: (via) => `Granted directly in the team's model list, and also via ${via}`,
  direct: "Granted directly in the team's model list",
  via: (via) => `Granted via ${via}`,
};

export function normalizeTeamModelSelection(models: string[] | undefined): string[] {
  return models && models.length > 0 ? models : [NO_DEFAULT_MODELS];
}

export function computeTeamModelBadges(
  models: string[],
  accessGroupModels: string[],
  accessGroupDetails: TeamAccessGroupModelGrant[] | undefined,
  copy: TeamModelBadgeCopy = DEFAULT_BADGE_COPY,
): TeamModelBadge[] {
  const grants = accessGroupDetails ?? [];
  const groupNamesFor = (model: string): string[] =>
    grants.filter((g) => g.models.includes(model)).map((g) => g.access_group_name);
  const viaGroups = (model: string): string => {
    const names = groupNamesFor(model);
    if (names.length === 0) return copy.fallbackAccessGroup;
    return names.length > 1 ? copy.accessGroups(names) : copy.accessGroup(names[0]);
  };

  const allProxy = models.length === 0 || models.includes(ALL_PROXY_MODELS);
  const directModels = allProxy ? [] : models.filter((m) => m !== NO_DEFAULT_MODELS);
  const groupModels = [...new Set(grants.length > 0 ? grants.flatMap((g) => g.models) : accessGroupModels)].filter(
    (m) => !directModels.includes(m),
  );

  const allProxyBadge: TeamModelBadge = {
    label: copy.allProxyModels,
    kind: "all-proxy",
    tooltip: models.includes(ALL_PROXY_MODELS) ? copy.allProxyEntryTooltip : copy.emptyModelListTooltip,
  };
  const noDefaultBadge: TeamModelBadge = {
    label: copy.noDefaultModels,
    kind: "no-default",
    tooltip: copy.noDefaultTooltip,
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
        tooltip: groupNamesFor(m).length > 0 ? copy.directAndVia(viaGroups(m)) : copy.direct,
      }),
    ),
    ...groupModels.map(
      (m): TeamModelBadge => ({
        label: m,
        kind: "access-group",
        tooltip: copy.via(viaGroups(m)),
      }),
    ),
  ];
}
