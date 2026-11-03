import type { TFunction } from "i18next";
import i18n from "@/i18n/i18n";
import type { RoutingGroup } from "./types";

export const groupNameByModel = (groups: RoutingGroup[], excludeGroupName?: string): Record<string, string> =>
  Object.fromEntries(
    groups
      .filter((group) => group.group_name !== excludeGroupName && group.routing_strategy !== "priority")
      .flatMap((group) => group.models.map((model) => [model, group.group_name] as const)),
  );

export const modelConflictError = (
  models: string[] | undefined,
  ownerByModel: Record<string, string>,
  t: TFunction = i18n.t.bind(i18n),
): string | null => {
  const conflicts = (models ?? []).filter((model) => Object.hasOwn(ownerByModel, model));
  if (conflicts.length === 0) return null;
  const detail = conflicts
    .map((model) => t("routingGroups.modelOwnership.claimedBy", { model, group: ownerByModel[model] }))
    .join(", ");
  return t("routingGroups.modelOwnership.conflict", { detail });
};
