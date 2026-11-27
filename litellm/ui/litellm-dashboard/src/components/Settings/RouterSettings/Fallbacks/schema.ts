import { z } from "zod";
import type { FallbackGroup } from "./FallbackGroupConfig";

type SelectedFallbackGroup = FallbackGroup & { primaryModel: string };

const isCompleteFallbackGroup = (group: FallbackGroup): group is SelectedFallbackGroup =>
  Boolean(group.primaryModel) && group.fallbackModels.length > 0;

export const incompleteFallbackGroups = (groups: FallbackGroup[]) =>
  groups.filter((group) => !isCompleteFallbackGroup(group));

export const addFallbacksSchema = z.object({
  groups: z.custom<FallbackGroup[]>().transform((groups, context) => {
    if (!groups.every(isCompleteFallbackGroup)) {
      context.addIssue({ code: "custom", message: "Incomplete fallback groups" });
      return z.NEVER;
    }
    return groups;
  }),
});

export const editFallbacksSchema = z.object({
  group: z.custom<FallbackGroup>().refine((group): group is SelectedFallbackGroup => Boolean(group.primaryModel)),
});
