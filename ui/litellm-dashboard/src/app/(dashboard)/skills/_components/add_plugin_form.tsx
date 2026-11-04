import React, { useState } from "react";
import type { ParseKeys } from "i18next";
import { CircleHelp } from "lucide-react";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { z } from "zod";
import { toast } from "@/lib/toast";
import { registerClaudeCodePlugin, uploadClaudeCodeSkill } from "@/components/networking";
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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
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

const MAX_SKILL_UPLOAD_BYTES = 10 * 1024 * 1024;

const addPluginShape = {
  sourceType: z.enum(["url", "upload"]),
  archiveFile: z
    .file()
    .refine((file) => file.name.toLowerCase().endsWith(".zip"), {
      error: () => i18n.t("skills.addForm.errors.zipRequired"),
    })
    .max(MAX_SKILL_UPLOAD_BYTES, { error: () => i18n.t("skills.addForm.errors.zipTooLarge") })
    .nullable(),
  skillUrl: z.string(),
  subPath: z.string(),
  sha256: z.string(),
  name: z
    .string()
    .min(1, { error: () => i18n.t("skills.addForm.errors.nameRequired") })
    .regex(/^[a-z0-9-]+$/, { error: () => i18n.t("skills.addForm.errors.nameInvalid") }),
  domain: z.string(),
  namespace: z.string(),
  description: z.string(),
  category: z.string().nullable(),
  keywords: z.string(),
  version: z.string(),
  authorName: z.string(),
  authorEmail: z.string().refine((value) => value === "" || z.email().safeParse(value).success, {
    error: () => i18n.t("skills.addForm.errors.emailInvalid"),
  }),
};

const addPluginSchema = z.object(addPluginShape).superRefine((values, context) => {
  if (values.sourceType === "upload") {
    if (!values.archiveFile) {
      context.addIssue({ code: "custom", path: ["archiveFile"], message: i18n.t("skills.addForm.errors.zipRequired") });
    }
    return;
  }
  if (!values.skillUrl) {
    context.addIssue({ code: "custom", path: ["skillUrl"], message: i18n.t("skills.addForm.errors.urlRequired") });
  }
  if (values.subPath && !isValidSubPath(values.subPath)) {
    context.addIssue({ code: "custom", path: ["subPath"], message: i18n.t("skills.addForm.errors.subPathInvalid") });
  }
  if (!isValidSha256(values.sha256)) {
    context.addIssue({ code: "custom", path: ["sha256"], message: i18n.t("skills.addForm.errors.sha256Invalid") });
  }
});

type AddPluginFormValues = z.infer<typeof addPluginSchema>;

const EMPTY_VALUES: AddPluginFormValues = {
  sourceType: "url",
  archiveFile: null,
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

const buildSkillMetadata = (values: AddPluginFormValues): Omit<SkillRegisterRequest, "source"> => {
  const author = buildAuthor(values);
  return {
    name: values.name.trim(),
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

const SUB_PATH_LOCK_REASON_KEY = {
  "git-subdir": "skills.addForm.subPathLock.gitSubdir",
  archive: "skills.addForm.subPathLock.archive",
} as const satisfies Record<string, ParseKeys>;

type SubPathLock = keyof typeof SUB_PATH_LOCK_REASON_KEY;

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
  const { t } = useTranslation();
  const form = useZodForm(addPluginSchema, { defaultValues: EMPTY_VALUES });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [urlPreview, setUrlPreview] = useState<SkillSourcePreview | null>(null);
  const [subPathLock, setSubPathLock] = useState<SubPathLock | null>(null);
  const [archiveInputKey, setArchiveInputKey] = useState(0);
  const isUpload = form.watch("sourceType") === "upload";

  const clearArchive = () => {
    form.setValue("archiveFile", null);
    form.clearErrors("archiveFile");
    setArchiveInputKey((key) => key + 1);
  };

  const resetDraft = () => {
    form.reset(EMPTY_VALUES);
    setArchiveInputKey((key) => key + 1);
    setUrlPreview(null);
    setSubPathLock(null);
  };

  const handleSourceChange = (value: unknown) => {
    if (value !== "url" && value !== "upload") return;
    form.setValue("sourceType", value);
    clearArchive();
    form.clearErrors(["skillUrl", "subPath", "sha256"]);
  };

  const selectArchive = (file: File | undefined) => {
    form.setValue("archiveFile", file ?? null, { shouldValidate: true });
    if (!file || !addPluginShape.archiveFile.safeParse(file).success) return;
    if (!form.getValues("name")) {
      form.setValue(
        "name",
        file.name
          .slice(0, -4)
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-+|-+$/g, ""),
      );
    }
  };

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
      toast.error(t("skills.addForm.toasts.noToken"));
      return;
    }

    if (values.sourceType === "url" && !urlPreview) {
      toast.error(t("skills.addForm.toasts.invalidUrl"));
      return;
    }

    if (!validatePluginName(values.name)) {
      toast.error(t("skills.addForm.toasts.invalidName"));
      return;
    }

    if (values.version && !isValidSemanticVersion(values.version)) {
      toast.error(t("skills.addForm.toasts.invalidVersion"));
      return;
    }

    if (values.authorEmail && !isValidEmail(values.authorEmail)) {
      toast.error(t("skills.addForm.toasts.invalidEmail"));
      return;
    }

    setIsSubmitting(true);
    try {
      const metadata = buildSkillMetadata(values);
      if (values.sourceType === "upload") {
        if (!values.archiveFile) return;
        await uploadClaudeCodeSkill(accessToken, values.archiveFile, metadata);
      } else if (urlPreview) {
        await registerClaudeCodePlugin(accessToken, {
          ...metadata,
          source: withArchiveDigest(urlPreview.parsed, values.sha256),
        });
      }
      toast.success(t("skills.addForm.toasts.registered"));
      resetDraft();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error registering skill:", error);
      toast.error(error instanceof Error && error.message ? error.message : t("skills.addForm.toasts.registerFailed"));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    if (isSubmitting) return;
    resetDraft();
    onClose();
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle>{t("skills.addForm.title")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={form.handleSubmit(handleSubmit)} noValidate className="mt-4">
            <FieldGroup>
              <FormField control={form.control} name="sourceType" label={t("skills.addForm.labels.source")}>
                {({ value }) => (
                  <RadioGroup
                    value={value}
                    disabled={isSubmitting}
                    onValueChange={handleSourceChange}
                    aria-label={t("skills.addForm.labels.source")}
                    className="flex gap-6"
                  >
                    <label className="flex items-center gap-2 text-sm">
                      <RadioGroupItem value="url" />
                      {t("skills.addForm.sourceOptions.url")}
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                      <RadioGroupItem value="upload" />
                      {t("skills.addForm.sourceOptions.upload")}
                    </label>
                  </RadioGroup>
                )}
              </FormField>

              {isUpload ? (
                <FormField
                  control={form.control}
                  name="archiveFile"
                  label={t("skills.addForm.labels.archive")}
                  description={t("skills.addForm.hints.archive")}
                >
                  {({ value, ...field }) => (
                    <>
                      <Input
                        {...field}
                        key={archiveInputKey}
                        type="file"
                        accept=".zip,application/zip"
                        disabled={isSubmitting}
                        onChange={(event) => selectArchive(event.target.files?.[0])}
                      />
                      {value && (
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="self-start"
                          onClick={clearArchive}
                          disabled={isSubmitting}
                        >
                          {t("skills.addForm.removeFile")}
                        </Button>
                      )}
                    </>
                  )}
                </FormField>
              ) : (
                <>
                  <FormField
                    control={form.control}
                    name="skillUrl"
                    label={labelWithHint(t("skills.addForm.labels.sourceUrl"), t("skills.addForm.hints.sourceUrl"))}
                  >
                    {({ ref, onChange, ...field }) => (
                      <Input
                        {...field}
                        ref={ref}
                        placeholder={t("skills.addForm.placeholders.sourceUrl")}
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
                    label={labelWithHint(t("skills.addForm.labels.subPath"), t("skills.addForm.hints.subPath"))}
                    description={subPathLock ? t(SUB_PATH_LOCK_REASON_KEY[subPathLock]) : undefined}
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
                      label={labelWithHint(t("skills.addForm.labels.sha256"), t("skills.addForm.hints.sha256"))}
                    >
                      {({ ref, ...field }) => (
                        <Input
                          {...field}
                          ref={ref}
                          placeholder={t("skills.addForm.placeholders.sha256")}
                          className="rounded-lg font-mono"
                        />
                      )}
                    </FormField>
                  )}

                  {urlPreview && (
                    <div className="rounded-lg border border-info/20 bg-info/10 px-3 py-2 text-sm text-info">
                      {t("skills.addForm.detected", { source: urlPreview.label })}
                    </div>
                  )}
                </>
              )}

              <FormField
                control={form.control}
                name="name"
                label={labelWithHint(t("skills.addForm.labels.name"), t("skills.addForm.hints.name"))}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="my-skill" className="rounded-lg" />}
              </FormField>

              <div className="flex gap-4">
                <FormField
                  control={form.control}
                  name="domain"
                  label={labelWithHint(t("skills.addForm.labels.domain"), t("skills.addForm.hints.domain"))}
                  className="flex-1"
                >
                  {({ ref, ...field }) => (
                    <Input
                      {...field}
                      ref={ref}
                      placeholder={t("skills.addForm.placeholders.domain")}
                      className="rounded-lg"
                    />
                  )}
                </FormField>
                <FormField
                  control={form.control}
                  name="namespace"
                  label={labelWithHint(t("skills.addForm.labels.namespace"), t("skills.addForm.hints.namespace"))}
                  className="flex-1"
                >
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="workflows" className="rounded-lg" />}
                </FormField>
              </div>

              <FormField
                control={form.control}
                name="description"
                label={labelWithHint(t("skills.addForm.labels.description"), t("skills.addForm.hints.description"))}
              >
                {({ ref, ...field }) => (
                  <Textarea
                    {...field}
                    ref={ref}
                    rows={3}
                    placeholder={t("skills.addForm.placeholders.description")}
                    maxLength={500}
                    className="rounded-lg"
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="category"
                label={labelWithHint(t("skills.addForm.labels.category"), t("skills.addForm.hints.category"))}
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <Combobox items={PREDEFINED_CATEGORIES} value={value} onValueChange={onChange}>
                    <ComboboxInput
                      id={id}
                      aria-invalid={ariaInvalid}
                      aria-describedby={ariaDescribedBy}
                      placeholder={t("skills.addForm.placeholders.category")}
                      className="w-full rounded-lg"
                      showClear={value != null && value !== ""}
                    />
                    <ComboboxContent>
                      <ComboboxEmpty>{t("skills.addForm.noCategories")}</ComboboxEmpty>
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
                label={labelWithHint(t("skills.addForm.labels.keywords"), t("skills.addForm.hints.keywords"))}
              >
                {({ ref, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    placeholder={t("skills.addForm.placeholders.keywords")}
                    className="rounded-lg"
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="version"
                label={labelWithHint(t("skills.addForm.labels.version"), t("skills.addForm.hints.version"))}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="1.0.0" className="rounded-lg" />}
              </FormField>

              <FormField
                control={form.control}
                name="authorName"
                label={labelWithHint(t("skills.addForm.labels.authorName"), t("skills.addForm.hints.authorName"))}
              >
                {({ ref, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    placeholder={t("skills.addForm.placeholders.authorName")}
                    className="rounded-lg"
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="authorEmail"
                label={labelWithHint(t("skills.addForm.labels.authorEmail"), t("skills.addForm.hints.authorEmail"))}
              >
                {({ ref, ...field }) => (
                  <Input {...field} ref={ref} type="email" placeholder="author@example.com" className="rounded-lg" />
                )}
              </FormField>
            </FieldGroup>

            <div className="mt-6 flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={handleCancel} disabled={isSubmitting}>
                {t("skills.addForm.cancel")}
              </Button>
              <Button type="submit" disabled={isSubmitting} aria-busy={isSubmitting}>
                {isSubmitting && <UiLoadingSpinner className="size-4" />}
                {isSubmitting ? t("skills.addForm.adding") : t("skills.addForm.submit")}
              </Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddPluginForm;
