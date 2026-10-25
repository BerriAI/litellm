import { TeamAccessGroupModelGrant } from "../team/teamModelAccess";
import { englishTranslate, type Translate } from "@/i18n/language";

export interface InheritedGrant {
  id: string;
  accessGroupNames: string[];
}

export function computeInheritedGrants(
  ids: string[] | undefined,
  grants: TeamAccessGroupModelGrant[] | undefined,
  idsOf: (grant: TeamAccessGroupModelGrant) => string[] | undefined,
): InheritedGrant[] {
  const known = grants ?? [];
  const allIds = [...new Set([...(ids ?? []), ...known.flatMap((grant) => idsOf(grant) ?? [])])];
  return allIds.map((id) => ({
    id,
    accessGroupNames: known
      .filter((grant) => (idsOf(grant) ?? []).includes(id))
      .map((grant) => grant.access_group_name),
  }));
}

export const inheritedGrantTooltip = (grant: InheritedGrant, t: Translate = englishTranslate): string => {
  // 组名属于业务数据，仅翻译来源说明，不对名称本身做匹配替换。
  if (grant.accessGroupNames.length === 0) {
    return t("Granted via an access group. Full ID: {id}", { id: grant.id });
  }
  return t(
    grant.accessGroupNames.length === 1
      ? "Granted via access group {groups}. Full ID: {id}"
      : "Granted via access groups {groups}. Full ID: {id}",
    {
      groups: grant.accessGroupNames.join(", "),
      id: grant.id,
    },
  );
};
