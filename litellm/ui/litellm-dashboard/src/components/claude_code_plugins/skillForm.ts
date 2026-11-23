import { z } from "zod";
import i18n from "@/i18n/i18n";
import type en from "@/i18n/locales/en.json";
import {
  isValidEmail,
  isValidSemanticVersion,
  isValidSha256,
  parseKeywords,
  parseSkillImportSource,
  parseSkillSource,
  requiresBrowserRefConfirmation,
  validatePluginName,
} from "./helpers";
import type {
  SkillHubSkill,
  SkillHubRegisterRequest,
  SkillImport,
  SkillMetadata,
  SkillUpdate,
  SkillUploadMetadata,
} from "./types";

type FormMessages = (typeof en)["skills"]["addForm"];
type FormErrorKey = `errors.${keyof FormMessages["errors"]}` | `toasts.${keyof FormMessages["toasts"]}`;

export interface SkillFormValues {
  mode: "upload" | "import" | "register" | "metadata" | "replace";
  archiveFile: File | null;
  url: string;
  skill: string;
  ref: string;
  path: string;
  name: string;
  domain: string;
  namespace: string;
  description: string;
  keywords: string;
  version: string;
  authorName: string;
  authorEmail: string;
  category: string;
  homepage: string;
  sha256: string;
}

const EMPTY_SKILL_FORM: SkillFormValues = {
  mode: "upload",
  archiveFile: null,
  url: "",
  skill: "",
  ref: "",
  path: "",
  name: "",
  domain: "",
  namespace: "",
  description: "",
  keywords: "",
  version: "",
  authorName: "",
  authorEmail: "",
  category: "",
  homepage: "",
  sha256: "",
};

export const skillFormValues = (skill?: SkillHubSkill): SkillFormValues => {
  if (!skill) return { ...EMPTY_SKILL_FORM };
  return {
    ...EMPTY_SKILL_FORM,
    mode: "metadata",
    name: skill.name,
    domain: skill.domain ?? "",
    namespace: skill.namespace ?? "",
    description: skill.description ?? "",
    keywords: skill.keywords?.join(", ") ?? "",
    authorName: skill.author?.name ?? "",
    authorEmail: skill.author?.email ?? "",
    category: skill.category ?? "",
    homepage: skill.homepage ?? "",
  };
};

export const buildSkillMetadata = (values: SkillFormValues, skill: SkillHubSkill): SkillMetadata => {
  const initial = skillFormValues(skill);
  return {
    version: values.version.trim() || null,
    expected_updated_at: skill.updated_at,
    ...(values.description !== initial.description ? { description: values.description.trim() || null } : {}),
    ...(values.domain !== initial.domain ? { domain: values.domain.trim() || null } : {}),
    ...(values.namespace !== initial.namespace ? { namespace: values.namespace.trim() || null } : {}),
    ...(values.keywords !== initial.keywords ? { keywords: parseKeywords(values.keywords) } : {}),
    ...(values.authorName !== initial.authorName || values.authorEmail !== initial.authorEmail
      ? {
          author: values.authorName.trim()
            ? { name: values.authorName.trim(), email: values.authorEmail.trim() || null }
            : null,
        }
      : {}),
    ...(values.category !== initial.category ? { category: values.category.trim() || null } : {}),
    ...(values.homepage !== initial.homepage ? { homepage: values.homepage.trim() || null } : {}),
  };
};

export const buildSkillUploadMetadata = (values: SkillFormValues): SkillUploadMetadata | undefined => {
  if (!values.name.trim()) return undefined;
  return {
    name: values.name.trim(),
    ...(values.version.trim() ? { version: values.version.trim() } : {}),
    ...(values.description.trim() ? { description: values.description.trim() } : {}),
    ...(values.domain.trim() ? { domain: values.domain.trim() } : {}),
    ...(values.namespace.trim() ? { namespace: values.namespace.trim() } : {}),
    ...(values.keywords.trim() ? { keywords: parseKeywords(values.keywords) } : {}),
    ...(values.authorName.trim()
      ? {
          author: {
            name: values.authorName.trim(),
            ...(values.authorEmail.trim() ? { email: values.authorEmail.trim() } : {}),
          },
        }
      : {}),
    ...(values.category.trim() ? { category: values.category.trim() } : {}),
    ...(values.homepage.trim() ? { homepage: values.homepage.trim() } : {}),
  };
};

export const buildSkillImport = (values: SkillFormValues): SkillImport | null => {
  const source = parseSkillImportSource(values.url, values.path, values.ref, values.skill);
  if (!source) return null;
  const metadata = buildSkillUploadMetadata(values);
  return {
    ...source.selection,
    ...(values.version.trim() ? { version: values.version.trim() } : {}),
    ...(metadata ? { metadata } : {}),
  };
};

export const buildSkillReference = (values: SkillFormValues): SkillHubRegisterRequest["source"] | null => {
  const preview = parseSkillSource(values.url, values.path, values.ref);
  if (!preview) return null;
  return {
    ...preview.parsed,
    ...(preview.parsed.source === "archive" && values.sha256.trim() ? { sha256: values.sha256.trim() } : {}),
  };
};

export const buildSkillRegistration = (values: SkillFormValues): SkillHubRegisterRequest | null => {
  const source = buildSkillReference(values);
  const metadata = buildSkillUploadMetadata(values);
  return source && metadata ? { ...metadata, source } : null;
};

export const buildSkillUpdate = (values: SkillFormValues, skill: SkillHubSkill): SkillUpdate => ({
  ...buildSkillMetadata(values, skill),
  ...(values.mode === "replace" ? { source: buildSkillReference(values) } : {}),
});

const CREATION_METADATA_FIELDS = [
  "description",
  "domain",
  "namespace",
  "keywords",
  "authorName",
  "authorEmail",
  "category",
  "homepage",
] as const;

type SkillFormIssue = (path: keyof SkillFormValues, key: FormErrorKey) => void;

function validateSkillSource(values: SkillFormValues, issue: SkillFormIssue) {
  if (values.mode === "upload") {
    if (!values.archiveFile?.name.toLowerCase().endsWith(".zip")) issue("archiveFile", "errors.zipRequired");
    if (values.archiveFile && values.archiveFile.size > 10 * 1024 * 1024) issue("archiveFile", "errors.zipTooLarge");
    return;
  }
  if (values.mode !== "metadata") validateSkillUrl(values, issue);
}

function validateSkillUrl(values: SkillFormValues, issue: SkillFormIssue) {
  if (values.mode === "import" && !values.url.trim()) return issue("url", "errors.urlRequired");
  const imported =
    values.mode === "import" ? parseSkillImportSource(values.url, values.path, values.ref, values.skill) : null;
  if (values.mode === "import" && !imported) return issue("url", "errors.importUrlInvalid");
  const preview = imported ? imported.preview : parseSkillSource(values.url, values.path, values.ref);
  if (!preview) {
    if (!imported) issue("url", "toasts.invalidUrl");
    return;
  }
  const source = preview.parsed;
  if (requiresBrowserRefConfirmation(preview, values.ref)) {
    issue("ref", "errors.browserRefRequired");
  }
  if (values.mode !== "import" && source.source === "archive" && !isValidSha256(values.sha256)) {
    issue("sha256", "errors.sha256Invalid");
  }
}

function validateRegistryName(values: SkillFormValues, issue: SkillFormIssue) {
  const needsName = values.mode === "register" || CREATION_METADATA_FIELDS.some((field) => values[field].trim());
  if (needsName && !values.name.trim()) issue("name", "errors.metadataNameRequired");
  if (values.name.trim() && !validatePluginName(values.name.trim())) issue("name", "errors.nameInvalid");
}

function validateSkillMetadata(values: SkillFormValues, skill: SkillHubSkill | undefined, issue: SkillFormIssue) {
  const initial = skillFormValues(skill);
  if (values.authorName === initial.authorName && values.authorEmail === initial.authorEmail) return;
  if (values.authorEmail.trim() && !values.authorName.trim()) issue("authorName", "errors.authorNameRequired");
  if (!isValidEmail(values.authorEmail.trim())) issue("authorEmail", "errors.emailInvalid");
}

const skillFormShape = {
  mode: z.enum(["upload", "import", "register", "metadata", "replace"]),
  archiveFile: z.file().nullable(),
  url: z.string(),
  skill: z.string(),
  ref: z.string(),
  path: z.string(),
  name: z.string(),
  domain: z.string(),
  namespace: z.string(),
  description: z.string(),
  keywords: z.string(),
  version: z.string(),
  authorName: z.string(),
  authorEmail: z.string(),
  category: z.string(),
  homepage: z.string(),
  sha256: z.string(),
};

export const skillFormSchema = (skill?: SkillHubSkill) =>
  z.object(skillFormShape).superRefine((values, context) => {
    const issue: SkillFormIssue = (path, key) =>
      context.addIssue({ code: "custom", path: [path], message: i18n.t(`skills.addForm.${key}`) });
    validateSkillSource(values, issue);
    if (values.version.trim() && !isValidSemanticVersion(values.version.trim()))
      issue("version", "toasts.invalidVersion");
    if (skill && values.mode === "upload") return;
    if (!skill) validateRegistryName(values, issue);
    validateSkillMetadata(values, skill, issue);
  });
