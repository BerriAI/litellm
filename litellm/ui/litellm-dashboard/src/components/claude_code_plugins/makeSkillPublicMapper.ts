import type { MakeSkillPublicValues } from "./makeSkillPublicSchema";
import type { Plugin } from "./types";

export const toSkillPublicationChanges = (
  { skillNames }: MakeSkillPublicValues,
  skills: readonly Pick<Plugin, "name" | "enabled">[],
) => {
  const selection = new Set(skillNames);
  return skills.flatMap(({ name, enabled }) => {
    const shouldBePublic = selection.has(name);
    return shouldBePublic === Boolean(enabled) ? [] : [{ name, enable: shouldBePublic }];
  });
};
