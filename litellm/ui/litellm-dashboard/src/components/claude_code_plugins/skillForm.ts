import { parseKeywords } from "./helpers";
import type { SkillHubSkill, SkillImport, SkillMetadata } from "./types";

export interface SkillFormValues {
  mode: "upload" | "import" | "metadata";
  archiveFile: File | null;
  url: string;
  skill: string;
  ref: string;
  path: string;
  domain: string;
  namespace: string;
  description: string;
  keywords: string;
  version: string;
}

export const skillFormValues = (skill?: SkillHubSkill): SkillFormValues => ({
  mode: skill ? "metadata" : "upload",
  archiveFile: null,
  url: "",
  skill: "",
  ref: "",
  path: "",
  domain: skill?.domain ?? "",
  namespace: skill?.namespace ?? "",
  description: skill?.description ?? "",
  keywords: skill?.keywords?.join(", ") ?? "",
  version: "",
});

export const buildSkillMetadata = (values: SkillFormValues, skill: SkillHubSkill): SkillMetadata => ({
  description: values.description.trim() || null,
  domain: values.domain.trim() || null,
  namespace: values.namespace.trim() || null,
  keywords:
    values.keywords === skillFormValues(skill).keywords ? skill.keywords ?? null : parseKeywords(values.keywords),
  version: values.version.trim() || null,
  expected_updated_at: skill.updated_at,
});

export const buildSkillImport = (values: SkillFormValues): SkillImport => ({
  url: values.url,
  ...(values.skill.trim() ? { skill: values.skill.trim() } : {}),
  ...(values.ref.trim() ? { ref: values.ref.trim() } : {}),
  ...(values.path.trim() ? { path: values.path.trim() } : {}),
  ...(values.version.trim() ? { version: values.version.trim() } : {}),
});
