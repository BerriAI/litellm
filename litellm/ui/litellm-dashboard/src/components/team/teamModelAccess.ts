import type { TFunction } from "i18next";

import i18n from "@/i18n/i18n";

import { unfurlWildcardModelsInList } from "../key_team_helpers/fetch_available_models_team_key";

export const ALL_PROXY_MODELS = "all-proxy-models";
export const NO_DEFAULT_MODELS = "no-default-models";

export interface TeamAccessGroupModelGrant {
  access_group_id: string;
  access_group_name: string;
  models: string[];
  mcp_server_ids?: string[];
  agent_ids?: string[];
  skill_names?: string[];
}

export type TeamModelBadgeKind = "all-proxy" | "no-default" | "direct" | "access-group";

export interface TeamModelBadge {
  label: string;
  kind: TeamModelBadgeKind;
  tooltip: string;
}

const ALL_TEAM_MODELS = "all-team-models";
const MODEL_SENTINELS: ReadonlySet<string> = new Set([ALL_PROXY_MODELS, ALL_TEAM_MODELS, NO_DEFAULT_MODELS]);

// Per-model rate limits and budgets are enforced on the model name a request sends,
// so only concrete names are offered: sentinels and wildcard patterns never match one.
export function teamPerModelLimitOptions(
  teamModels: string[],
  accessGroupModels: string[],
  userModels: string[],
): string[] {
  const coversAllModels =
    teamModels.length === 0 || teamModels.includes(ALL_PROXY_MODELS) || teamModels.includes(ALL_TEAM_MODELS);
  const granted = coversAllModels ? userModels : [...new Set([...teamModels, ...accessGroupModels])];
  return unfurlWildcardModelsInList(granted, userModels).filter(
    (model) => !MODEL_SENTINELS.has(model) && !model.endsWith("*"),
  );
}

export function normalizeTeamModelSelection(models: string[] | undefined): string[] {
  return models && models.length > 0 ? models : [NO_DEFAULT_MODELS];
}

export const describeGroups = (names: string[], t: TFunction = i18n.t.bind(i18n)): string =>
  t("teams.modelAccess.groups", { count: names.length, names: names.join(", ") });

export function computeTeamModelBadges(
  models: string[],
  accessGroupModels: string[],
  accessGroupDetails: TeamAccessGroupModelGrant[] | undefined,
  t: TFunction = i18n.t.bind(i18n),
): TeamModelBadge[] {
  const grants = accessGroupDetails ?? [];
  const groupNamesFor = (model: string): string[] =>
    grants.filter((g) => g.models.includes(model)).map((g) => g.access_group_name);
  const viaGroups = (model: string): string => {
    const names = groupNamesFor(model);
    return names.length > 0 ? describeGroups(names, t) : t("teams.modelAccess.anAccessGroup");
  };

  const allProxy = models.length === 0 || models.includes(ALL_PROXY_MODELS);
  const directModels = allProxy ? [] : models.filter((m) => m !== NO_DEFAULT_MODELS);
  const groupModels = [...new Set(grants.length > 0 ? grants.flatMap((g) => g.models) : accessGroupModels)].filter(
    (m) => !directModels.includes(m),
  );

  const allProxyBadge: TeamModelBadge = {
    label: t("teams.modelAccess.allProxy.label"),
    kind: "all-proxy",
    tooltip: models.includes(ALL_PROXY_MODELS)
      ? t("teams.modelAccess.allProxy.sentinelTooltip")
      : t("teams.modelAccess.allProxy.emptyListTooltip"),
  };
  const noDefaultBadge: TeamModelBadge = {
    label: t("teams.modelAccess.noDefault.label"),
    kind: "no-default",
    tooltip: t("teams.modelAccess.noDefault.tooltip"),
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
            ? t("teams.modelAccess.direct.alsoViaGroupsTooltip", { groups: viaGroups(m) })
            : t("teams.modelAccess.direct.tooltip"),
      }),
    ),
    ...groupModels.map(
      (m): TeamModelBadge => ({
        label: m,
        kind: "access-group",
        tooltip: t("teams.modelAccess.viaGroupsTooltip", { groups: viaGroups(m) }),
      }),
    ),
  ];
}
