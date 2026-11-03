"use client";

import React, { useState, useEffect } from "react";
import { Trans, useTranslation } from "react-i18next";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { z } from "zod";
import { fetchUserModels } from "@/components/organisms/create_key_button";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { tagInfoCall, tagUpdateCall } from "@/components/networking";
import { Tag } from "@/components/tag_management/types";
import { toast } from "@/lib/toast";
import NumericalInput from "@/components/shared/numerical_input";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import i18n from "@/i18n/i18n";
import { useZodForm } from "@/lib/forms/useZodForm";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CheckIcon, ChevronRight, CopyIcon } from "lucide-react";

const tagEditShape = {
  name: z.string().min(1, { error: () => i18n.t("tagManagement.editForm.validation.nameRequired") }),
  description: z.string().optional(),
  models: z.array(z.string()).optional(),
  max_budget: z
    .union([z.string(), z.number()])
    .refine((value) => value === "" || (Number.isFinite(Number(value)) && Number(value) >= 0), {
      error: () => i18n.t("tagManagement.editForm.validation.budgetNonnegative"),
    })
    .optional(),
  budget_duration: z.string().nullish(),
};

const tagEditSchema = z.object(tagEditShape);

type TagEditFormValues = z.output<typeof tagEditSchema>;

const budgetPayload = (value: TagEditFormValues["max_budget"]): number | null | undefined => {
  if (value === undefined) return undefined;
  if (value === "") return null;
  return Number(value);
};

interface TagEditFormProps {
  tag: Tag;
  seedBudgetFields: boolean;
  userModels: string[];
  onCancel: () => void;
  onSave: (values: TagEditFormValues) => Promise<void>;
}

const TagEditForm: React.FC<TagEditFormProps> = ({ tag, seedBudgetFields, userModels, onCancel, onSave }) => {
  const { t } = useTranslation();
  const [budgetSectionOpen, setBudgetSectionOpen] = useState(false);
  const form = useZodForm(tagEditSchema, {
    defaultValues: {
      name: tag.name,
      description: tag.description,
      models: tag.models,
      max_budget: seedBudgetFields ? tag.litellm_budget_table?.max_budget : undefined,
      budget_duration: seedBudgetFields ? tag.litellm_budget_table?.budget_duration : undefined,
    },
  });

  const submitVisibleValues = (values: TagEditFormValues): Promise<void> =>
    onSave(budgetSectionOpen ? values : { ...values, max_budget: undefined, budget_duration: undefined });

  const modelOptions = userModels.map((modelId) => ({ label: getModelDisplayName(modelId), value: modelId }));

  return (
    <form onSubmit={form.handleSubmit(submitVisibleValues)} noValidate>
      <FieldGroup>
        <FormField control={form.control} name="name" label={t("tagManagement.editForm.fields.name")}>
          {({ ref, ...field }) => <Input {...field} ref={ref} />}
        </FormField>

        <FormField control={form.control} name="description" label={t("tagManagement.editForm.fields.description")}>
          {({ ref, value, ...field }) => <Textarea {...field} ref={ref} value={value ?? ""} rows={4} />}
        </FormField>

        <FormField
          control={form.control}
          name="models"
          label={t("tagManagement.editForm.fields.models")}
          description={t("tagManagement.editForm.fields.modelsDescription")}
        >
          {({ value, onChange }) => (
            <MultiSelect
              options={modelOptions}
              value={value}
              onValueChange={onChange}
              placeholder={t("tagManagement.editForm.fields.modelsPlaceholder")}
            />
          )}
        </FormField>
      </FieldGroup>

      <Collapsible
        open={budgetSectionOpen}
        onOpenChange={setBudgetSectionOpen}
        className="mt-4 mb-4 rounded-md border border-border"
      >
        <CollapsibleTrigger className="group flex w-full items-center justify-between px-4 py-3 text-base font-medium text-foreground">
          {t("tagManagement.editForm.budgetSection")}
          <ChevronRight className="size-4 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
        </CollapsibleTrigger>
        <CollapsibleContent className="px-4 pb-4">
          <FieldGroup className="mt-4">
            <FormField
              control={form.control}
              name="max_budget"
              label={t("tagManagement.editForm.fields.maxBudget")}
              description={t("tagManagement.editForm.fields.maxBudgetDescription")}
            >
              {({ ref, value, ...field }) => <NumericalInput {...field} value={value ?? ""} step={0.01} />}
            </FormField>

            <FormField
              control={form.control}
              name="budget_duration"
              label={t("tagManagement.editForm.fields.resetBudget")}
              description={t("tagManagement.editForm.fields.resetBudgetDescription")}
            >
              {({ id, value, onChange }) => (
                <BudgetDurationDropdown id={id} value={value ?? null} onChange={onChange} />
              )}
            </FormField>
          </FieldGroup>

          <div className="mt-4 rounded-md border border-border bg-muted p-3">
            <p className="text-sm text-muted-foreground">
              <Trans
                i18nKey="tagManagement.editForm.rateLimitsNotice"
                components={{
                  issue: (
                    <a
                      href="https://github.com/BerriAI/litellm/issues/new"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-info underline hover:text-info/80"
                    />
                  ),
                }}
              />
            </p>
          </div>
        </CollapsibleContent>
      </Collapsible>

      <div className="flex justify-end space-x-2">
        <Button type="button" variant="outline" onClick={onCancel}>
          {t("tagManagement.editForm.cancel")}
        </Button>
        <Button type="submit">{t("tagManagement.editForm.save")}</Button>
      </div>
    </form>
  );
};

interface TagInfoViewProps {
  tagId: string;
  onClose: () => void;
  accessToken: string | null;
  is_admin: boolean;
  editTag: boolean;
}

const TagInfoView: React.FC<TagInfoViewProps> = ({ tagId, onClose, accessToken, is_admin, editTag }) => {
  const { t } = useTranslation();
  const [tagDetails, setTagDetails] = useState<Tag | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(editTag);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const fetchTagDetails = async () => {
    if (!accessToken) return;
    try {
      const response = await tagInfoCall(accessToken, [tagId]);
      const tagData = response[tagId];
      if (tagData) {
        setTagDetails(tagData);
      }
    } catch (error) {
      console.error("Error fetching tag details:", error);
      toast.fromError(t("tagManagement.info.toasts.fetchFailed", { error: String(error) }));
    }
  };

  useEffect(() => {
    fetchTagDetails();
  }, [tagId, accessToken]);

  useEffect(() => {
    if (accessToken) {
      // Using dummy values for userID and userRole since they're required by the function
      // TODO: Pass these as props if needed for the actual API implementation
      fetchUserModels("dummy-user", "Admin", accessToken, setUserModels);
    }
  }, [accessToken]);

  const handleSave = async (values: TagEditFormValues) => {
    if (!accessToken) return;
    try {
      await tagUpdateCall(accessToken, {
        name: values.name,
        description: values.description,
        models: values.models,
        max_budget: budgetPayload(values.max_budget),
        tpm_limit: undefined,
        rpm_limit: undefined,
        budget_duration: values.budget_duration,
      });
      toast.success(t("tagManagement.info.toasts.updated"));
      setIsEditing(false);
      fetchTagDetails();
    } catch (error) {
      console.error("Error updating tag:", error);
      toast.fromError(t("tagManagement.info.toasts.updateFailed", { error: String(error) }));
    }
  };

  if (!tagDetails) {
    return <div>{t("tagManagement.info.loading")}</div>;
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">
            {t("tagManagement.info.back")}
          </Button>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">{t("tagManagement.info.tagNameLabel")}</span>
            <span className="font-mono px-2 py-1 bg-muted rounded-sm text-sm border border-border">
              {tagDetails.name}
            </span>
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => copyToClipboard(tagDetails.name, "tag-name")}
              className={`transition-all duration-200 ${
                copiedStates["tag-name"]
                  ? "text-success bg-success/10 border-success/20"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
            >
              {copiedStates["tag-name"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
            </Button>
          </div>
          <p className="text-sm text-muted-foreground">
            {tagDetails.description || t("tagManagement.info.noDescription")}
          </p>
        </div>
        {is_admin && !isEditing && (
          <Button onClick={() => setIsEditing(true)}>{t("tagManagement.info.editTag")}</Button>
        )}
      </div>

      {isEditing ? (
        <Card>
          <CardContent>
            <TagEditForm
              tag={tagDetails}
              seedBudgetFields={editTag}
              userModels={userModels}
              onCancel={() => setIsEditing(false)}
              onSave={handleSave}
            />
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card>
            <CardContent>
              <CardTitle>{t("tagManagement.info.details.title")}</CardTitle>
              <div className="space-y-4 mt-4">
                <div>
                  <p className="font-medium">{t("tagManagement.info.details.name")}</p>
                  <p>{tagDetails.name}</p>
                </div>
                <div>
                  <p className="font-medium">{t("tagManagement.info.details.description")}</p>
                  <p>{tagDetails.description || "-"}</p>
                </div>
                <div>
                  <p className="font-medium">{t("tagManagement.info.details.models")}</p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {!tagDetails.models || tagDetails.models.length === 0 ? (
                      <Badge variant="secondary">{t("tagManagement.info.details.allModels")}</Badge>
                    ) : (
                      tagDetails.models.map((modelId) => (
                        <Badge key={modelId} variant="secondary">
                          <SimpleTooltip content={t("tagManagement.info.details.modelId", { id: modelId })}>
                            {tagDetails.model_info?.[modelId] || modelId}
                          </SimpleTooltip>
                        </Badge>
                      ))
                    )}
                  </div>
                </div>
                <div>
                  <p className="font-medium">{t("tagManagement.info.details.created")}</p>
                  <p>{tagDetails.created_at ? new Date(tagDetails.created_at).toLocaleString() : "-"}</p>
                </div>
                <div>
                  <p className="font-medium">{t("tagManagement.info.details.lastUpdated")}</p>
                  <p>{tagDetails.updated_at ? new Date(tagDetails.updated_at).toLocaleString() : "-"}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {tagDetails.litellm_budget_table && (
            <Card>
              <CardContent>
                <CardTitle>{t("tagManagement.info.budget.title")}</CardTitle>
                <div className="space-y-4 mt-4">
                  {tagDetails.litellm_budget_table.max_budget !== undefined &&
                    tagDetails.litellm_budget_table.max_budget !== null && (
                      <div>
                        <p className="font-medium">{t("tagManagement.info.budget.maxBudget")}</p>
                        <p>${tagDetails.litellm_budget_table.max_budget}</p>
                      </div>
                    )}
                  {tagDetails.litellm_budget_table.budget_duration && (
                    <div>
                      <p className="font-medium">{t("tagManagement.info.budget.duration")}</p>
                      <p>{tagDetails.litellm_budget_table.budget_duration}</p>
                    </div>
                  )}
                  {tagDetails.litellm_budget_table.tpm_limit !== undefined &&
                    tagDetails.litellm_budget_table.tpm_limit !== null && (
                      <div>
                        <p className="font-medium">{t("tagManagement.info.budget.tpmLimit")}</p>
                        <p>{tagDetails.litellm_budget_table.tpm_limit.toLocaleString()}</p>
                      </div>
                    )}
                  {tagDetails.litellm_budget_table.rpm_limit !== undefined &&
                    tagDetails.litellm_budget_table.rpm_limit !== null && (
                      <div>
                        <p className="font-medium">{t("tagManagement.info.budget.rpmLimit")}</p>
                        <p>{tagDetails.litellm_budget_table.rpm_limit.toLocaleString()}</p>
                      </div>
                    )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};

export default TagInfoView;
