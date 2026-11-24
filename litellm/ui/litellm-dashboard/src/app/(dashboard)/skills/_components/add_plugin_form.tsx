import { useState } from "react";
import type { UseFormReturn } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { toast } from "@/lib/toast";
import { ApiError } from "@/lib/http/client";
import {
  editSkillSubmission,
  uploadSkillSubmission,
  importSkillHubSkill,
  updateSkillHubSkill,
  uploadSkillHubSkill,
} from "@/components/networking";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Textarea } from "@/components/ui/textarea";
import { useZodForm } from "@/lib/forms/useZodForm";
import {
  availableSkillUrlModes,
  parseSkillImportSource,
  parseSkillSource,
  requiresBrowserRefConfirmation,
} from "@/components/claude_code_plugins/helpers";
import {
  isPendingSkill,
  type SkillHubSkill,
  type SkillMutationResponse,
  type SkillSubmission,
} from "@/components/claude_code_plugins/types";
import {
  buildSkillImport,
  buildSkillMetadata,
  buildSkillRegistration,
  buildSkillUpdate,
  buildSkillUploadMetadata,
  skillFormSchema,
  skillFormValues,
  type SkillFormValues,
} from "@/components/claude_code_plugins/skillForm";

const EDIT_SKILL_SOURCES = ["metadata", "upload", "replace"] as const;
const SUBMISSION_SOURCES = ["metadata", "upload"] as const;
const NEW_SKILL_SOURCES = ["upload", "register"] as const;
const URL_SKILL_SOURCES = ["register", "import"] as const;

const ADVANCED_FIELDS = [
  "authorName",
  "authorEmail",
  "category",
  "domain",
  "namespace",
  "keywords",
  "homepage",
] as const;

interface AddPluginFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess?: (response: SkillMutationResponse) => void;
  skill?: SkillHubSkill;
  submission?: { value: SkillSubmission; publishedVersion: string | null; onChanged: () => void };
  requiresApproval?: boolean;
}

function formLabels(editing: boolean, submission: boolean, requiresApproval: boolean) {
  if (submission) return { title: "skills.submissions.edit", submit: "skills.submissions.save" } as const;
  const directSubmit = editing ? "skills.addForm.save" : "skills.addForm.submit";
  return {
    title: editing ? "skills.addForm.editTitle" : "skills.addForm.title",
    submit: requiresApproval ? "skills.addForm.submitReview" : directSubmit,
  } as const;
}

interface SkillFormFieldsProps {
  form: UseFormReturn<SkillFormValues>;
  skill?: SkillHubSkill;
}

interface SkillSourceSelectorsProps {
  form: UseFormReturn<SkillFormValues>;
  mode: "register" | "import" | "replace";
}

function SkillSourceSelectors({ form, mode }: SkillSourceSelectorsProps) {
  const { t } = useTranslation();
  const sourceUrl = form.watch("url");
  const sourcePath = form.watch("path");
  const sourceRef = form.watch("ref");
  const source =
    mode === "import"
      ? parseSkillImportSource(sourceUrl, sourcePath, sourceRef)
      : { preview: parseSkillSource(sourceUrl, sourcePath, sourceRef), skillFromUrl: undefined };
  const sourcePreview = source?.preview;
  const needsRef = requiresBrowserRefConfirmation(sourcePreview, sourceRef);
  const displayPreview = needsRef ? null : sourcePreview;
  const isArchive = sourcePreview?.parsed.source === "archive";
  const archivePathLock = isArchive ? "archive" : undefined;
  const pathLock = sourcePreview?.pathFromUrl ? "gitSubdir" : archivePathLock;
  const referenceFields = isArchive ? (["path"] as const) : (["ref", "path"] as const);
  const sourceFields = mode === "import" ? (["skill", "ref", "path"] as const) : referenceFields;
  const lockedValues: Partial<Record<keyof SkillFormValues, string>> = {
    ...(pathLock ? { path: displayPreview?.parsed.path ?? "" } : {}),
    ...(source?.skillFromUrl ? { skill: source.skillFromUrl } : {}),
  };
  const sourceDescriptions: Partial<Record<keyof SkillFormValues, string>> = {
    skill: t(source?.skillFromUrl ? "skills.addForm.hints.skillFromUrl" : "skills.addForm.hints.skill"),
    path: t(pathLock ? `skills.addForm.subPathLock.${pathLock}` : "skills.addForm.hints.subPath"),
    ref: t(needsRef ? "skills.addForm.errors.browserRefRequired" : "skills.addForm.hints.ref"),
  };
  return (
    <>
      {sourceFields.map((name) => (
        <FormField
          key={name}
          control={form.control}
          name={name}
          label={t(`skills.addForm.neutralLabels.${name}`)}
          description={sourceDescriptions[name]}
        >
          {(field) => (
            <Input
              {...field}
              disabled={lockedValues[name] !== undefined}
              value={lockedValues[name] ?? field.value}
              placeholder={name === "ref" ? displayPreview?.parsed.ref ?? displayPreview?.parsed.sha : undefined}
            />
          )}
        </FormField>
      ))}
      <p className="text-sm text-muted-foreground">{t(`skills.addForm.${mode}Hint`)}</p>
      {displayPreview && (
        <p className="text-sm text-muted-foreground">
          {t("skills.addForm.detected", { source: displayPreview.label })}
        </p>
      )}
    </>
  );
}

function SkillSourceFields({
  form,
  skill,
  changeMode,
}: SkillFormFieldsProps & { changeMode: (value: string) => void }) {
  const { t } = useTranslation();
  const mode = form.watch("mode");
  const sourceUrl = form.watch("url");
  const urlModes = availableSkillUrlModes(sourceUrl);
  const changeSourceMode = (url: string) => {
    if (skill) return;
    const available = availableSkillUrlModes(url);
    if (available.length === 1 && available[0] !== mode) changeMode(available[0]);
  };
  if (mode === "metadata") return null;
  if (mode === "upload") {
    return (
      <FormField
        key="archiveFile"
        control={form.control}
        name="archiveFile"
        label={t("skills.addForm.labels.archive")}
        description={t(skill ? "skills.addForm.replacementHint" : "skills.addForm.uploadHint")}
      >
        {({ value, onChange, ...field }) => (
          <Input
            {...field}
            type="file"
            accept=".zip,application/zip"
            onChange={(event) => onChange(event.target.files?.[0] ?? null)}
          />
        )}
      </FormField>
    );
  }
  return (
    <>
      <FormField key="url" control={form.control} name="url" label={t("skills.addForm.neutralLabels.url")}>
        {(field) => (
          <Input
            {...field}
            onChange={(event) => {
              field.onChange(event);
              changeSourceMode(event.target.value);
            }}
          />
        )}
      </FormField>
      {!skill && (
        <FormField
          control={form.control}
          name="mode"
          label={t("skills.addForm.urlStorage")}
          description={t("skills.addForm.urlStorageHint")}
        >
          {({ value }) => (
            <RadioGroup
              value={value}
              onValueChange={changeMode}
              aria-label={t("skills.addForm.urlStorage")}
              className="gap-3"
            >
              {URL_SKILL_SOURCES.map((option) => (
                <label key={option} className="flex items-center gap-2 text-sm has-[[aria-disabled=true]]:opacity-50">
                  <RadioGroupItem value={option} disabled={!urlModes.some((mode) => mode === option)} />
                  {t(`skills.addForm.urlModes.${option}`)}
                </label>
              ))}
            </RadioGroup>
          )}
        </FormField>
      )}
      <SkillSourceSelectors form={form} mode={mode} />
    </>
  );
}

function SkillMetadataFields({ form, skill }: SkillFormFieldsProps) {
  const { t } = useTranslation();
  const mode = form.watch("mode");
  const sourceUrl = form.watch("url");
  const sourcePath = form.watch("path");
  const sourceRef = form.watch("ref");
  if (skill && mode === "upload") return null;
  const referenceMode = mode === "register" || mode === "replace";
  const sourcePreview = referenceMode ? parseSkillSource(sourceUrl, sourcePath, sourceRef) : null;
  const hasAdvancedErrors =
    ADVANCED_FIELDS.some((field) => form.formState.errors[field]) || Boolean(form.formState.errors.sha256);
  return (
    <>
      {!skill && (
        <FormField
          control={form.control}
          name="name"
          label={t(mode === "register" ? "skills.addForm.registryName" : "skills.addForm.optionalRegistryName")}
          description={t("skills.addForm.registryNameHint")}
        >
          {(field) => <Input {...field} />}
        </FormField>
      )}
      <FormField control={form.control} name="description" label={t("skills.addForm.labels.description")}>
        {(field) => <Textarea {...field} />}
      </FormField>
      <details open={hasAdvancedErrors || undefined} className="space-y-4">
        <summary className="cursor-pointer text-sm font-medium">{t("skills.addForm.advancedMetadata")}</summary>
        <p className="text-sm text-muted-foreground">{t("skills.addForm.authorOverrideHint")}</p>
        <FieldGroup>
          {ADVANCED_FIELDS.map((name) => (
            <FormField
              key={name}
              control={form.control}
              name={name}
              label={t(
                name === "domain" || name === "namespace" || name === "keywords"
                  ? `skills.addForm.neutralLabels.${name}`
                  : `skills.addForm.labels.${name}`,
              )}
            >
              {(field) => <Input {...field} />}
            </FormField>
          ))}
          {sourcePreview?.parsed.source === "archive" && (
            <FormField control={form.control} name="sha256" label={t("skills.addForm.labels.sha256")}>
              {(field) => <Input {...field} />}
            </FormField>
          )}
        </FieldGroup>
      </details>
    </>
  );
}

function SkillVersionField({ form, skill, submission }: SkillFormFieldsProps & Pick<AddPluginFormProps, "submission">) {
  const { t } = useTranslation();
  const version = submission ? submission.publishedVersion : skill?.version;
  return (
    <FormField
      control={form.control}
      name="version"
      label={t(skill ? "skills.addForm.versionOverride" : "skills.addForm.labels.version")}
      description={
        skill ? t("skills.addForm.versionContext", { version: version ?? t("skills.submissions.notSet") }) : undefined
      }
    >
      {(field) => <Input {...field} placeholder={skill ? t("skills.addForm.automaticVersion") : "1.0.0"} />}
    </FormField>
  );
}

export default function AddPluginForm({
  visible,
  onClose,
  accessToken,
  onSuccess,
  skill,
  submission,
  requiresApproval = false,
}: AddPluginFormProps) {
  const { t } = useTranslation();
  const form = useZodForm(skillFormSchema(skill), {
    defaultValues: {
      ...skillFormValues(skill),
      ...(submission ? { version: submission.value.proposal.version ?? "" } : {}),
    },
  });
  const [error, setError] = useState<Error | null>(null);
  const conflictedSubmission = error instanceof ApiError && error.status === 409 ? submission : undefined;
  const isSubmitting = form.formState.isSubmitting;
  const submit = async (values: SkillFormValues) => {
    if (!accessToken) return;
    setError(null);
    try {
      if (submission && skill) {
        if (values.mode === "upload" && values.archiveFile) {
          await uploadSkillSubmission(accessToken, submission.value, values.archiveFile, values.version);
        } else {
          await editSkillSubmission(accessToken, submission.value.id, {
            ...buildSkillMetadata(values, skill),
            expected_revision: submission.value.revision ?? 0,
          });
        }
        toast.success(t("skills.submissions.saved"));
        submission.onChanged();
        onClose();
        return;
      }
      const write = () => {
        if (values.mode === "upload" && values.archiveFile) {
          return uploadSkillHubSkill(accessToken, values.archiveFile, values.version, {
            skill,
            metadata: skill ? undefined : buildSkillUploadMetadata(values),
          });
        }
        if (skill) return updateSkillHubSkill(accessToken, skill.name, buildSkillUpdate(values, skill));
        if (values.mode === "register") {
          const registration = buildSkillRegistration(values);
          if (!registration) return;
          return importSkillHubSkill(accessToken, registration);
        }
        const imported = buildSkillImport(values);
        if (imported) return importSkillHubSkill(accessToken, imported);
      };
      const response = await write();
      if (!response) return;
      const publishedMessage = skill ? "skills.addForm.toasts.updated" : "skills.addForm.toasts.registered";
      toast.success(t(isPendingSkill(response) ? "skills.addForm.toasts.submitted" : publishedMessage));
      onSuccess?.(response);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err : new Error(t("skills.addForm.toasts.registerFailed")));
    }
  };
  const changeMode = (value: string) => {
    const nextMode = [...EDIT_SKILL_SOURCES, ...URL_SKILL_SOURCES].find((option) => option === value);
    if (!nextMode) return;
    form.setValue("mode", nextMode === "register" ? availableSkillUrlModes(form.getValues("url"))[0] : nextMode);
    form.setValue("archiveFile", null);
    form.clearErrors();
    setError(null);
  };
  const labels = formLabels(Boolean(skill), Boolean(submission), requiresApproval);
  const editSources = submission ? SUBMISSION_SOURCES : EDIT_SKILL_SOURCES;
  const sources = skill ? editSources : NEW_SKILL_SOURCES;
  return (
    <Dialog open={visible} onOpenChange={(open) => !open && !isSubmitting && onClose()}>
      <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t(labels.title)}</DialogTitle>
        </DialogHeader>
        {skill && <p className="text-sm font-medium">{skill.name}</p>}
        {requiresApproval && <p className="text-sm text-muted-foreground">{t("skills.addForm.approvalNotice")}</p>}
        <form onSubmit={form.handleSubmit(submit)} className="space-y-6">
          <fieldset disabled={isSubmitting} className="space-y-6">
            <FormField control={form.control} name="mode" label={t("skills.addForm.labels.source")}>
              {({ value }) => (
                <RadioGroup
                  value={value === "import" ? "register" : value}
                  onValueChange={changeMode}
                  aria-label={t("skills.addForm.labels.source")}
                  className="flex flex-wrap gap-5"
                >
                  {sources.map((option) => (
                    <label key={option} className="flex items-center gap-2 text-sm">
                      <RadioGroupItem value={option} />
                      {t(`skills.addForm.modes.${option}`)}
                    </label>
                  ))}
                </RadioGroup>
              )}
            </FormField>
            <FieldGroup>
              <SkillSourceFields form={form} skill={skill} changeMode={changeMode} />
              <SkillMetadataFields form={form} skill={skill} />
              <SkillVersionField form={form} skill={skill} submission={submission} />
            </FieldGroup>
          </fieldset>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error.message}
            </p>
          )}
          <div className="flex justify-end gap-2">
            {conflictedSubmission && (
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  conflictedSubmission.onChanged();
                  onClose();
                }}
                disabled={isSubmitting}
              >
                {t("skills.submissions.reload")}
              </Button>
            )}
            <Button type="button" variant="outline" onClick={onClose} disabled={isSubmitting}>
              {t("skills.addForm.cancel")}
            </Button>
            <Button type="submit" disabled={isSubmitting} aria-busy={isSubmitting}>
              {t(isSubmitting ? "skills.addForm.saving" : labels.submit)}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
