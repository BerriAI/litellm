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
): string | null => {
  const conflicts = (models ?? []).filter((model) => Object.hasOwn(ownerByModel, model));
  if (conflicts.length === 0) return null;
  const detail = conflicts.map((model) => `${model} (in "${ownerByModel[model]}")`).join(", ");
  return `Each model may belong to at most one non-priority group. Already claimed: ${detail}`;
};
