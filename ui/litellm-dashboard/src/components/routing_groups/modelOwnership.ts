import type { RoutingGroup } from "./types";
import { englishTranslate, type Translate } from "@/i18n/language";

export const groupNameByModel = (groups: RoutingGroup[], excludeGroupName?: string): Record<string, string> =>
  Object.fromEntries(
    groups
      .filter((group) => group.group_name !== excludeGroupName && group.routing_strategy !== "priority")
      .flatMap((group) => group.models.map((model) => [model, group.group_name] as const)),
  );

export const modelConflictError = (
  models: string[] | undefined,
  ownerByModel: Record<string, string>,
  t: Translate = englishTranslate,
): string | null => {
  const conflicts = (models ?? []).filter((model) => Object.hasOwn(ownerByModel, model));
  if (conflicts.length === 0) return null;
  // 名称只作为占位符插入，不把资源名当作词典键
  const detail = conflicts
    .map((model) => t('{model} (in "{group}")', { model, group: ownerByModel[model] }))
    .join(", ");
  return t("Each model may belong to at most one non-priority group. Already claimed: {detail}", { detail });
};
