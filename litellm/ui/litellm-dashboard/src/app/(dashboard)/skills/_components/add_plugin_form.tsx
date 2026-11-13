import { useState } from "react";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import i18n from "@/i18n/i18n";
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
import { isValidSemanticVersion } from "@/components/claude_code_plugins/helpers";
import {
  isPendingSkill,
  type SkillHubSkill,
  type SkillMutationResponse,
  type SkillSubmission,
} from "@/components/claude_code_plugins/types";
import { buildSkillImport, buildSkillMetadata, skillFormValues } from "@/components/claude_code_plugins/skillForm";

const EDIT_SKILL_SOURCES = ["metadata", "upload"] as const;
const NEW_SKILL_SOURCES = ["upload", "import"] as const;

interface AddPluginFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess?: (response: SkillMutationResponse) => void;
  skill?: SkillHubSkill;
  submission?: { value: SkillSubmission; publishedVersion: string | null; onChanged: () => void };
  requiresApproval?: boolean;
}

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
  domain: z.string(),
  namespace: z.string(),
  description: z.string(),
  keywords: z.string(),
  version: z.string().refine((value) => !value.trim() || isValidSemanticVersion(value.trim()), {
    error: () => i18n.t("skills.addForm.toasts.invalidVersion"),
  }),
};
const skillSchema = z.object(skillShape).superRefine((values, context) => {
  if (values.mode === "upload" && !values.archiveFile) {
    context.addIssue({ code: "custom", path: ["archiveFile"], message: i18n.t("skills.addForm.errors.zipRequired") });
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

function formLabels(editing: boolean, submission: boolean, requiresApproval: boolean) {
  if (submission) return { title: "skills.submissions.edit", submit: "skills.submissions.save" } as const;
  const directSubmit = editing ? "skills.addForm.save" : "skills.addForm.submit";
  return {
    title: editing ? "skills.addForm.editTitle" : "skills.addForm.title",
    submit: requiresApproval ? "skills.addForm.submitReview" : directSubmit,
  } as const;
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
  const form = useZodForm(skillSchema, {
    defaultValues: {
      ...skillFormValues(skill),
      ...(submission ? { version: submission.value.proposal.version ?? "" } : {}),
    },
  });
  const mode = form.watch("mode");
  const [error, setError] = useState<Error | null>(null);
  const conflictedSubmission = error instanceof ApiError && error.status === 409 ? submission : undefined;
  const isSubmitting = form.formState.isSubmitting;
  const submit = async (values: z.infer<typeof skillSchema>) => {
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
          return uploadSkillHubSkill(accessToken, values.archiveFile, values.version, skill);
        }
        if (skill) return updateSkillHubSkill(accessToken, skill.name, buildSkillMetadata(values, skill));
        return importSkillHubSkill(accessToken, buildSkillImport(values));
      };
      const response = await write();
      const publishedMessage = skill ? "skills.addForm.toasts.updated" : "skills.addForm.toasts.registered";
      toast.success(t(isPendingSkill(response) ? "skills.addForm.toasts.submitted" : publishedMessage));
      onSuccess?.(response);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err : new Error(t("skills.addForm.toasts.registerFailed")));
    }
  };
  const changeMode = (value: string) => {
    if (value !== "upload" && value !== "import" && value !== "metadata") return;
    form.setValue("mode", value);
    form.setValue("archiveFile", null);
    form.clearErrors();
    setError(null);
  };
  const labels = formLabels(Boolean(skill), Boolean(submission), requiresApproval);
  const fields =
    mode === "import" ? (["url", "skill", "ref", "path"] as const) : (["domain", "namespace", "keywords"] as const);
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
                  {({ value, onChange, ...field }) => (
                    <Input
                      {...field}
                      type="file"
                      accept=".zip,application/zip"
                      onChange={(event) => onChange(event.target.files?.[0] ?? null)}
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
                    ? t("skills.addForm.versionContext", {
                        version:
                          (submission ? submission.publishedVersion : skill.version) ?? t("skills.submissions.notSet"),
                      })
                    : undefined
                }
              >
                {(field) => <Input {...field} placeholder={skill ? t("skills.addForm.automaticVersion") : "1.0.0"} />}
              </FormField>
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
