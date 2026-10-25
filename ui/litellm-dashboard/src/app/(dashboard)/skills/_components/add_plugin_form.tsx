import { useLanguage } from "@/contexts/LanguageContext";
import React, { useState } from "react";
import TranslatedMessage from "@/components/shared/TranslatedMessage";
import { CircleHelp } from "lucide-react";
import { z } from "zod";
import { toast } from "@/lib/toast";
import { registerClaudeCodePlugin } from "@/components/networking";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";
import {
  validatePluginName,
  isValidSemanticVersion,
  isValidEmail,
  parseKeywords,
  parseSkillSource,
  isValidSubPath,
  isValidSha256,
  SkillSourcePreview,
} from "@/components/claude_code_plugins/helpers";
import { PluginAuthor, PluginSource, SkillRegisterRequest } from "@/components/claude_code_plugins/types";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface AddPluginFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

// 只在字段渲染时翻译本地校验，已显示的错误也能随语言切换。
const addPluginShape = {
  skillUrl: z.string().min(1, "Please enter a repository or zip archive URL"),
  subPath: z
    .string()
    .refine(
      (value) => !value || isValidSubPath(value),
      "Subfolder must be a relative path like plugins/my-skill (letters, numbers, dots, hyphens, underscores)",
    ),
  sha256: z.string().refine(isValidSha256, "SHA-256 must be a 64-character hex digest"),
  name: z
    .string()
    .min(1, "Please enter skill name")
    .regex(/^[a-z0-9-]+$/, "Name must be kebab-case (lowercase, numbers, hyphens only)"),
  domain: z.string(),
  namespace: z.string(),
  description: z.string(),
  category: z.string().nullable(),
  keywords: z.string(),
  version: z.string(),
  authorName: z.string(),
  authorEmail: z
    .string()
    .refine((value) => value === "" || z.email().safeParse(value).success, "Please enter a valid email"),
};
const addPluginSchema = z.object(addPluginShape);

type AddPluginFormValues = z.infer<typeof addPluginSchema>;

const EMPTY_VALUES: AddPluginFormValues = {
  skillUrl: "",
  subPath: "",
  sha256: "",
  name: "",
  domain: "",
  namespace: "",
  description: "",
  category: null,
  keywords: "",
  version: "",
  authorName: "",
  authorEmail: "",
};

const buildAuthor = (values: AddPluginFormValues): PluginAuthor | undefined => {
  const name = values.authorName.trim();
  const email = values.authorEmail.trim();
  if (!name) {
    return undefined;
  }
  return email ? { name, email } : { name };
};

const archiveUrlOf = (preview: SkillSourcePreview | null): string | undefined =>
  preview?.parsed.source === "archive" ? preview.parsed.url : undefined;

const withArchiveDigest = (source: PluginSource, sha256: string): PluginSource => {
  const digest = sha256.trim();
  return source.source === "archive" && digest ? { ...source, sha256: digest.toLowerCase() } : source;
};

const buildRegisterRequest = (values: AddPluginFormValues, source: PluginSource): SkillRegisterRequest => {
  const author = buildAuthor(values);
  return {
    name: values.name.trim(),
    source: withArchiveDigest(source, values.sha256),
    ...(values.version ? { version: values.version.trim() } : {}),
    ...(values.description ? { description: values.description.trim() } : {}),
    ...(author ? { author } : {}),
    ...(values.category ? { category: values.category } : {}),
    ...(values.keywords ? { keywords: parseKeywords(values.keywords) } : {}),
    ...(values.domain ? { domain: values.domain.trim() } : {}),
    ...(values.namespace ? { namespace: values.namespace.trim() } : {}),
  };
};

const PREDEFINED_CATEGORIES = [
  "Development",
  "Productivity",
  "Learning",
  "Security",
  "Data & Analytics",
  "Integration",
  "Testing",
  "Documentation",
];

const SUB_PATH_LOCK_REASON = {
  "git-subdir": "The URL already points to a subfolder, so this field is disabled",
  archive: "A zip archive is installed as a whole, so this field is disabled",
} as const;

type SubPathLock = keyof typeof SUB_PATH_LOCK_REASON;

const subPathLockFor = (source: PluginSource["source"] | undefined): SubPathLock | null =>
  source === "git-subdir" || source === "archive" ? source : null;

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const AddPluginForm: React.FC<AddPluginFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const { t } = useLanguage();
  const form = useZodForm(addPluginSchema, { defaultValues: EMPTY_VALUES });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [urlPreview, setUrlPreview] = useState<SkillSourcePreview | null>(null);
  const [subPathLock, setSubPathLock] = useState<SubPathLock | null>(null);

  const recomputePreview = (skillUrl: string, subPath: string) => {
    const lock = subPathLockFor(parseSkillSource(skillUrl)?.parsed.source);
    setSubPathLock(lock);
    if (lock && form.getValues("subPath")) {
      form.setValue("subPath", "");
    }
    const preview = parseSkillSource(skillUrl, lock ? undefined : subPath);
    if (archiveUrlOf(preview) !== archiveUrlOf(urlPreview) && form.getValues("sha256")) {
      form.setValue("sha256", "");
    }
    setUrlPreview(preview);
    if (preview && !form.getValues("name")) {
      form.setValue("name", preview.suggestedName);
    }
  };

  const handleSubmit = async (values: AddPluginFormValues) => {
    if (!accessToken) {
      toast.error(t("No access token available"));
      return;
    }

    if (!urlPreview) {
      toast.error(t("Please enter a valid repository or zip archive URL"));
      return;
    }

    if (!validatePluginName(values.name)) {
      toast.error(t("Skill name must be kebab-case (lowercase letters, numbers, and hyphens only)"));
      return;
    }

    if (values.version && !isValidSemanticVersion(values.version)) {
      toast.error(t("Version must be in semantic versioning format (e.g., 1.0.0)"));
      return;
    }

    if (values.authorEmail && !isValidEmail(values.authorEmail)) {
      toast.error(t("Invalid email format"));
      return;
    }

    setIsSubmitting(true);
    try {
      await registerClaudeCodePlugin(accessToken, buildRegisterRequest(values, urlPreview.parsed));
      toast.success(<TranslatedMessage message="Skill registered successfully" />);
      form.reset(EMPTY_VALUES);
      setUrlPreview(null);
      setSubPathLock(null);
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error registering skill:", error);
      toast.error(
        error instanceof Error && error.message ? (
          error.message
        ) : (
          <TranslatedMessage message="Failed to register skill" />
        ),
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    form.reset(EMPTY_VALUES);
    setUrlPreview(null);
    setSubPathLock(null);
    onClose();
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle>{t("Add New Skill")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={form.handleSubmit(handleSubmit)} noValidate className="mt-4">
            <FieldGroup>
              <FormField
                control={form.control}
                name="skillUrl"
                renderError={({ message }) => message && t(message)}
                label={labelWithHint(
                  t("Source URL"),
                  t(
                    "Paste an HTTPS git repository URL from GitHub, GitLab, Bitbucket, or a self-hosted host (e.g. github.com/org/repo or github.com/org/repo/tree/main/my-skill), or an HTTPS link to a .zip archive of the skill hosted on S3 or any static file server. For a private repository use its SSH clone URL (git@ghe.example.com:org/repo.git) so Claude Code clones it with your own SSH key.",
                  ),
                )}
              >
                {({ ref, onChange, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    placeholder="https://github.com/org/repo or https://bucket.s3.amazonaws.com/my-skill.zip"
                    className="rounded-lg"
                    onChange={(event) => {
                      onChange(event);
                      recomputePreview(event.target.value, form.getValues("subPath"));
                    }}
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="subPath"
                renderError={({ message }) => message && t(message)}
                label={labelWithHint(
                  t("Subfolder path (Optional)"),
                  t(
                    "Path within the repository where the skill lives (e.g., plugins/my-skill). Leave empty if the skill is at the repo root.",
                  ),
                )}
                description={subPathLock ? t(SUB_PATH_LOCK_REASON[subPathLock]) : undefined}
              >
                {({ ref, onChange, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    placeholder="plugins/my-skill"
                    className="rounded-lg"
                    onChange={(event) => {
                      onChange(event);
                      recomputePreview(form.getValues("skillUrl"), event.target.value);
                    }}
                    disabled={subPathLock !== null}
                  />
                )}
              </FormField>

              {urlPreview?.parsed.source === "archive" && (
                <FormField
                  control={form.control}
                  name="sha256"
                  renderError={({ message }) => message && t(message)}
                  label={labelWithHint(
                    t("Archive SHA-256 (Optional)"),
                    t(
                      "Hex digest of the zip file. Claude Code refuses to install the archive if its checksum does not match.",
                    ),
                  )}
                >
                  {({ ref, ...field }) => (
                    <Input {...field} ref={ref} placeholder={t("64 hex characters")} className="rounded-lg font-mono" />
                  )}
                </FormField>
              )}

              {urlPreview && (
                <div className="rounded-lg border border-info/20 bg-info/10 px-3 py-2 text-sm text-info">
                  {t("Detected:")} {urlPreview.label}
                </div>
              )}

              <FormField
                control={form.control}
                name="name"
                renderError={({ message }) => message && t(message)}
                label={labelWithHint(t("Skill Name"), t("Unique identifier in kebab-case format (e.g., my-skill)"))}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="my-skill" className="rounded-lg" />}
              </FormField>

              <div className="flex gap-4">
                <FormField
                  control={form.control}
                  name="domain"
                  label={labelWithHint(
                    t("Domain (Optional)"),
                    t("Top-level grouping in the Skill Hub (e.g., Productivity)"),
                  )}
                  className="flex-1"
                >
                  {({ ref, ...field }) => (
                    <Input {...field} ref={ref} placeholder={t("Productivity")} className="rounded-lg" />
                  )}
                </FormField>
                <FormField
                  control={form.control}
                  name="namespace"
                  label={labelWithHint(t("Namespace (Optional)"), t("Sub-grouping within domain (e.g., workflows)"))}
                  className="flex-1"
                >
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="workflows" className="rounded-lg" />}
                </FormField>
              </div>

              <FormField
                control={form.control}
                name="description"
                label={labelWithHint(t("Description (Optional)"), t("Brief description of what the skill does"))}
              >
                {({ ref, ...field }) => (
                  <Textarea
                    {...field}
                    ref={ref}
                    rows={3}
                    placeholder={t("A skill that helps with...")}
                    maxLength={500}
                    className="rounded-lg"
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="category"
                label={labelWithHint(t("Category (Optional)"), t("Select a category or enter a custom one"))}
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <Combobox items={PREDEFINED_CATEGORIES} value={value} onValueChange={onChange}>
                    <ComboboxInput
                      id={id}
                      aria-invalid={ariaInvalid}
                      aria-describedby={ariaDescribedBy}
                      placeholder={t("Select or type a category")}
                      className="w-full rounded-lg"
                      showClear={value != null && value !== ""}
                    />
                    <ComboboxContent>
                      <ComboboxEmpty>{t("No matching categories")}</ComboboxEmpty>
                      <ComboboxList>
                        {(category: string) => (
                          <ComboboxItem key={category} value={category}>
                            {category}
                          </ComboboxItem>
                        )}
                      </ComboboxList>
                    </ComboboxContent>
                  </Combobox>
                )}
              </FormField>

              <FormField
                control={form.control}
                name="keywords"
                label={labelWithHint(t("Keywords (Optional)"), t("Comma-separated list of keywords for search"))}
              >
                {({ ref, ...field }) => (
                  <Input {...field} ref={ref} placeholder="search, web, api" className="rounded-lg" />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="version"
                label={labelWithHint(t("Version (Optional)"), t("Semantic version (e.g., 1.0.0)"))}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="1.0.0" className="rounded-lg" />}
              </FormField>

              <FormField
                control={form.control}
                name="authorName"
                label={labelWithHint(t("Author Name (Optional)"), t("Name of the skill author or organization"))}
              >
                {({ ref, ...field }) => (
                  <Input {...field} ref={ref} placeholder={t("Your Name or Organization")} className="rounded-lg" />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="authorEmail"
                renderError={({ message }) => message && t(message)}
                label={labelWithHint(t("Author Email (Optional)"), t("Contact email for the skill author"))}
              >
                {({ ref, ...field }) => (
                  <Input {...field} ref={ref} type="email" placeholder="author@example.com" className="rounded-lg" />
                )}
              </FormField>
            </FieldGroup>

            <div className="mt-6 flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={handleCancel} disabled={isSubmitting}>
                {t("Cancel")}
              </Button>
              <Button type="submit" disabled={isSubmitting} aria-busy={isSubmitting}>
                {isSubmitting && <UiLoadingSpinner className="size-4" />}
                {isSubmitting ? t("Adding...") : t("Add Skill")}
              </Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddPluginForm;
