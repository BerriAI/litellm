import { useState } from "react";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import i18n from "@/i18n/i18n";
import { toast } from "@/lib/toast";
import { importSkillHubSkill, updateSkillHubSkill, uploadSkillHubSkill } from "@/components/networking";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
<<<<<<< HEAD
=======
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
import { Switch } from "@/components/ui/switch";
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
>>>>>>> cf3abb938a (feat(claude-code): host plugin zips for Claude Desktop marketplaces)
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Textarea } from "@/components/ui/textarea";
import { useZodForm } from "@/lib/forms/useZodForm";
import { isValidSemanticVersion } from "@/components/claude_code_plugins/helpers";
import { isPendingSkill, type SkillHubSkill, type SkillMutationResponse } from "@/components/claude_code_plugins/types";
import { buildSkillImport, buildSkillMetadata, skillFormValues } from "@/components/claude_code_plugins/skillForm";

const EDIT_SKILL_SOURCES = ["metadata", "upload"] as const;
const NEW_SKILL_SOURCES = ["upload", "import"] as const;

interface AddPluginFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: (response: SkillMutationResponse) => void;
  skill?: SkillHubSkill;
  requiresApproval?: boolean;
}

<<<<<<< HEAD
const skillShape = {
  mode: z.enum(["upload", "import", "metadata"]),
  archiveFile: z
    .file()
    .refine((file) => file.name.toLowerCase().endsWith(".zip"), {
      error: () => i18n.t("skills.addForm.errors.zipRequired"),
    })
    .max(10 * 1024 * 1024, { error: () => i18n.t("skills.addForm.errors.zipTooLarge") })
    .nullable(),
  url: z.string(),
  skill: z.string(),
  ref: z.string(),
  path: z.string(),
=======
const addPluginShape = {
  skillUrl: z.string().min(1, "Please enter a repository or zip archive URL"),
  subPath: z
    .string()
    .refine(
      (value) => !value || isValidSubPath(value),
      "Subfolder must be a relative path like plugins/my-skill (letters, numbers, dots, hyphens, underscores)",
    ),
  sha256: z.string().refine(isValidSha256, "SHA-256 must be a 64-character hex digest"),
  hostArchive: z.boolean(),
  name: z
    .string()
    .min(1, "Please enter skill name")
    .regex(/^[a-z0-9-]+$/, "Name must be kebab-case (lowercase, numbers, hyphens only)"),
>>>>>>> cf3abb938a (feat(claude-code): host plugin zips for Claude Desktop marketplaces)
  domain: z.string(),
  namespace: z.string(),
  description: z.string(),
  keywords: z.string(),
  version: z.string().refine((value) => !value.trim() || isValidSemanticVersion(value.trim()), {
    error: () => i18n.t("skills.addForm.toasts.invalidVersion"),
  }),
};
<<<<<<< HEAD
const skillSchema = z.object(skillShape).superRefine((values, context) => {
  if (values.mode === "upload" && !values.archiveFile) {
    context.addIssue({ code: "custom", path: ["archiveFile"], message: i18n.t("skills.addForm.errors.zipRequired") });
=======

const addPluginSchema = z.object(addPluginShape);

type AddPluginFormValues = z.infer<typeof addPluginSchema>;

const EMPTY_VALUES: AddPluginFormValues = {
  skillUrl: "",
  subPath: "",
  sha256: "",
  hostArchive: false,
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
>>>>>>> cf3abb938a (feat(claude-code): host plugin zips for Claude Desktop marketplaces)
  }
  if (values.mode === "import" && !values.url.trim()) {
    context.addIssue({ code: "custom", path: ["url"], message: i18n.t("skills.addForm.errors.urlRequired") });
  }
  if (values.mode === "metadata" && values.description.length > 1024) {
    context.addIssue({
      code: "custom",
      path: ["description"],
      message: i18n.t("skills.addForm.errors.descriptionTooLong"),
    });
  }
});

<<<<<<< HEAD
export default function AddPluginForm({
  visible,
  onClose,
  accessToken,
  onSuccess,
  skill,
  requiresApproval = false,
}: AddPluginFormProps) {
  const { t } = useTranslation();
  const form = useZodForm(skillSchema, { defaultValues: skillFormValues(skill) });
  const mode = form.watch("mode");
  const [error, setError] = useState<string | null>(null);
  const isSubmitting = form.formState.isSubmitting;
  const submit = async (values: z.infer<typeof skillSchema>) => {
    if (!accessToken) return;
    setError(null);
=======
const archiveUrlOf = (preview: SkillSourcePreview | null): string | undefined =>
  preview?.parsed.source === "archive" ? preview.parsed.url : undefined;

const hostsArchiveByDefault = (source: PluginSource["source"] | undefined): boolean => source === "archive";

const withArchiveDigest = (source: PluginSource, sha256: string): PluginSource => {
  const digest = sha256.trim();
  return source.source === "archive" && digest ? { ...source, sha256: digest.toLowerCase() } : source;
};

const buildRegisterRequest = (values: AddPluginFormValues, source: PluginSource): SkillRegisterRequest => {
  const author = buildAuthor(values);
  const serveArchive = values.hostArchive !== hostsArchiveByDefault(source.source) ? values.hostArchive : undefined;
  return {
    name: values.name.trim(),
    source: withArchiveDigest(source, values.sha256),
    ...(serveArchive !== undefined ? { serve_archive: serveArchive } : {}),
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
    const nextSource = preview?.parsed.source;
    if (nextSource !== urlPreview?.parsed.source) {
      form.setValue("hostArchive", hostsArchiveByDefault(nextSource));
    }
    setUrlPreview(preview);
    if (preview && !form.getValues("name")) {
      form.setValue("name", preview.suggestedName);
    }
  };

  const handleSubmit = async (values: AddPluginFormValues) => {
    if (!accessToken) {
      toast.error("No access token available");
      return;
    }

    if (!urlPreview) {
      toast.error("Please enter a valid repository or zip archive URL");
      return;
    }

    if (!validatePluginName(values.name)) {
      toast.error("Skill name must be kebab-case (lowercase letters, numbers, and hyphens only)");
      return;
    }

    if (values.version && !isValidSemanticVersion(values.version)) {
      toast.error("Version must be in semantic versioning format (e.g., 1.0.0)");
      return;
    }

    if (values.authorEmail && !isValidEmail(values.authorEmail)) {
      toast.error("Invalid email format");
      return;
    }

    setIsSubmitting(true);
>>>>>>> cf3abb938a (feat(claude-code): host plugin zips for Claude Desktop marketplaces)
    try {
      const write = () => {
        if (values.mode === "upload" && values.archiveFile) {
          return uploadSkillHubSkill(accessToken, values.archiveFile, values.version, skill);
        }
        if (skill) return updateSkillHubSkill(accessToken, skill.name, buildSkillMetadata(values, skill));
        return importSkillHubSkill(accessToken, buildSkillImport(values));
      };
      const response = await write();
      const publishedMessage = skill ? "skills.addForm.toasts.updated" : "skills.addForm.toasts.registered";
      toast.success(t(isPendingSkill(response) ? "skills.addForm.toasts.submitted" : publishedMessage));
      onSuccess(response);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("skills.addForm.toasts.registerFailed"));
    }
  };
  const changeMode = (value: string) => {
    if (value !== "upload" && value !== "import" && value !== "metadata") return;
    form.setValue("mode", value);
    form.setValue("archiveFile", null);
    form.clearErrors();
    setError(null);
  };
  const directLabel = skill ? "skills.addForm.save" : "skills.addForm.submit";
  const submitLabel = requiresApproval ? "skills.addForm.submitReview" : directLabel;
  const fields =
    mode === "import" ? (["url", "skill", "ref", "path"] as const) : (["domain", "namespace", "keywords"] as const);
  return (
    <Dialog open={visible} onOpenChange={(open) => !open && !isSubmitting && onClose()}>
      <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t(skill ? "skills.addForm.editTitle" : "skills.addForm.title")}</DialogTitle>
        </DialogHeader>
        {skill && <p className="text-sm font-medium">{skill.name}</p>}
        {requiresApproval && <p className="text-sm text-muted-foreground">{t("skills.addForm.approvalNotice")}</p>}
        <form onSubmit={form.handleSubmit(submit)} className="space-y-6">
          <fieldset disabled={isSubmitting} className="space-y-6">
            <FormField control={form.control} name="mode" label={t("skills.addForm.labels.source")}>
              {({ value }) => (
                <RadioGroup value={value} onValueChange={changeMode} className="flex gap-5">
                  {(skill ? EDIT_SKILL_SOURCES : NEW_SKILL_SOURCES).map((option) => (
                    <label key={option} className="flex items-center gap-2 text-sm">
                      <RadioGroupItem value={option} />
                      {t(`skills.addForm.modes.${option}`)}
                    </label>
                  ))}
                </RadioGroup>
              )}
            </FormField>
            <FieldGroup>
              {mode === "upload" ? (
                <FormField
                  control={form.control}
                  name="archiveFile"
                  label={t("skills.addForm.labels.archive")}
                  description={t(skill ? "skills.addForm.replacementHint" : "skills.addForm.uploadHint")}
                >
<<<<<<< HEAD
                  {({ value, onChange, ...field }) => (
                    <Input
                      {...field}
                      type="file"
                      accept=".zip,application/zip"
                      onChange={(event) => onChange(event.target.files?.[0] ?? null)}
=======
                  {({ ref, ...field }) => (
                    <Input {...field} ref={ref} placeholder="64 hex characters" className="rounded-lg font-mono" />
                  )}
                </FormField>
              )}

              {urlPreview && (
                <FormField
                  control={form.control}
                  name="hostArchive"
                  orientation="horizontal"
                  label={labelWithHint(
                    "Host for Claude Desktop",
                    "LiteLLM downloads the plugin when you add it, stores it as a zip on this server and serves it from the same origin as the marketplace, which is the only plugin source Claude Desktop installs from a marketplace URL. Claude Code keeps working either way. Git sources must be hosted on GitHub.",
                  )}
                  description="Serve this skill to Claude Desktop as a zip archive from this LiteLLM server"
                >
                  {({ id, value, onChange, "aria-describedby": ariaDescribedBy }) => (
                    <Switch id={id} checked={value} onCheckedChange={onChange} aria-describedby={ariaDescribedBy} />
                  )}
                </FormField>
              )}

              {urlPreview && (
                <div className="rounded-lg border border-info/20 bg-info/10 px-3 py-2 text-sm text-info">
                  Detected: {urlPreview.label}
                </div>
              )}

              <FormField
                control={form.control}
                name="name"
                label={labelWithHint("Skill Name", "Unique identifier in kebab-case format (e.g., my-skill)")}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="my-skill" className="rounded-lg" />}
              </FormField>

              <div className="flex gap-4">
                <FormField
                  control={form.control}
                  name="domain"
                  label={labelWithHint("Domain (Optional)", "Top-level grouping in the Skill Hub (e.g., Productivity)")}
                  className="flex-1"
                >
                  {({ ref, ...field }) => (
                    <Input {...field} ref={ref} placeholder="Productivity" className="rounded-lg" />
                  )}
                </FormField>
                <FormField
                  control={form.control}
                  name="namespace"
                  label={labelWithHint("Namespace (Optional)", "Sub-grouping within domain (e.g., workflows)")}
                  className="flex-1"
                >
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="workflows" className="rounded-lg" />}
                </FormField>
              </div>

              <FormField
                control={form.control}
                name="description"
                label={labelWithHint("Description (Optional)", "Brief description of what the skill does")}
              >
                {({ ref, ...field }) => (
                  <Textarea
                    {...field}
                    ref={ref}
                    rows={3}
                    placeholder="A skill that helps with..."
                    maxLength={500}
                    className="rounded-lg"
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="category"
                label={labelWithHint("Category (Optional)", "Select a category or enter a custom one")}
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <Combobox items={PREDEFINED_CATEGORIES} value={value} onValueChange={onChange}>
                    <ComboboxInput
                      id={id}
                      aria-invalid={ariaInvalid}
                      aria-describedby={ariaDescribedBy}
                      placeholder="Select or type a category"
                      className="w-full rounded-lg"
                      showClear={value != null && value !== ""}
>>>>>>> cf3abb938a (feat(claude-code): host plugin zips for Claude Desktop marketplaces)
                    />
                  )}
                </FormField>
              ) : (
                <>
                  {mode === "metadata" && (
                    <FormField control={form.control} name="description" label={t("skills.addForm.labels.description")}>
                      {(field) => <Textarea {...field} maxLength={1024} />}
                    </FormField>
                  )}
                  {fields.map((name) => (
                    <FormField
                      key={name}
                      control={form.control}
                      name={name}
                      label={t(`skills.addForm.neutralLabels.${name}`)}
                    >
                      {(field) => <Input {...field} />}
                    </FormField>
                  ))}
                  {mode === "import" && (
                    <p className="text-sm text-muted-foreground">{t("skills.addForm.importHint")}</p>
                  )}
                </>
              )}
              <FormField
                control={form.control}
                name="version"
                label={t(skill ? "skills.addForm.versionOverride" : "skills.addForm.labels.version")}
                description={
                  skill
                    ? t("skills.addForm.versionContext", { version: skill.version ?? t("skills.submissions.notSet") })
                    : undefined
                }
              >
                {(field) => <Input {...field} placeholder={skill ? t("skills.addForm.automaticVersion") : "1.0.0"} />}
              </FormField>
            </FieldGroup>
          </fieldset>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={onClose} disabled={isSubmitting}>
              {t("skills.addForm.cancel")}
            </Button>
            <Button type="submit" disabled={isSubmitting} aria-busy={isSubmitting}>
              {t(isSubmitting ? "skills.addForm.saving" : submitLabel)}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
