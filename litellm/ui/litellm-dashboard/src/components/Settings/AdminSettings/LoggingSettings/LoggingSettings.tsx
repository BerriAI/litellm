"use client";

import {
  ConfigType,
  GeneralSettingsFieldName,
  useDeleteProxyConfigField,
  useProxyConfig,
} from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { useStoreRequestInSpendLogs } from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CircleHelp, Clock } from "lucide-react";
import type { ParseKeys } from "i18next";
import React, { useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { countDisplay, formToPayload, omittedFieldNames, storedValuesToForm } from "./mapper";
import { loggingSettingsSchema, type LoggingSettingsFormValues } from "./schema";

const STORE_PROMPTS_FIELD_NAME = "store_prompts_in_spend_logs";

interface OptionalField {
  readonly name: GeneralSettingsFieldName;
  readonly kind: "duration" | "count";
  readonly label: ParseKeys;
  readonly placeholder: ParseKeys;
  readonly fallbackTooltip: ParseKeys;
}

const OPTIONAL_FIELDS: readonly OptionalField[] = [
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    kind: "duration",
    label: "adminPanel.loggingSettings.fields.retentionPeriod.label",
    placeholder: "adminPanel.loggingSettings.fields.retentionPeriod.placeholder",
    fallbackTooltip: "adminPanel.loggingSettings.fields.retentionPeriod.hint",
  },
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_CLEANUP_BATCH_SIZE,
    kind: "count",
    label: "adminPanel.loggingSettings.fields.cleanupBatchSize.label",
    placeholder: "adminPanel.loggingSettings.fields.cleanupBatchSize.placeholder",
    fallbackTooltip: "adminPanel.loggingSettings.fields.cleanupBatchSize.hint",
  },
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_CLEANUP_MAX_BATCHES,
    kind: "count",
    label: "adminPanel.loggingSettings.fields.cleanupMaxBatches.label",
    placeholder: "adminPanel.loggingSettings.fields.cleanupMaxBatches.placeholder",
    fallbackTooltip: "adminPanel.loggingSettings.fields.cleanupMaxBatches.hint",
  },
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_CLEANUP_RUN_BUDGET,
    kind: "duration",
    label: "adminPanel.loggingSettings.fields.cleanupRunBudget.label",
    placeholder: "adminPanel.loggingSettings.fields.cleanupRunBudget.placeholder",
    fallbackTooltip: "adminPanel.loggingSettings.fields.cleanupRunBudget.hint",
  },
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_CLEANUP_BATCH_TIMEOUT,
    kind: "duration",
    label: "adminPanel.loggingSettings.fields.cleanupBatchTimeout.label",
    placeholder: "adminPanel.loggingSettings.fields.cleanupBatchTimeout.placeholder",
    fallbackTooltip: "adminPanel.loggingSettings.fields.cleanupBatchTimeout.hint",
  },
];

const SKIP_PASSED_FIELD_NAME = "guardrail_log_skip_passed";

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

interface LoggingSettingsFormProps {
  initialValues: LoggingSettingsFormValues;
  describeField: (name: string, fallback: string) => string;
  skipPassedEditable: boolean;
  isSaving: boolean;
  onSubmit: (formValues: LoggingSettingsFormValues) => void;
}

const LoggingSettingsForm: React.FC<LoggingSettingsFormProps> = ({
  initialValues,
  describeField,
  skipPassedEditable,
  isSaving,
  onSubmit,
}) => {
  const { t } = useTranslation();
  const form = useZodForm(loggingSettingsSchema, { defaultValues: initialValues });

  return (
    <TooltipProvider>
      <form onSubmit={form.handleSubmit(onSubmit)} noValidate>
        <FieldGroup>
          <FormField
            control={form.control}
            name={STORE_PROMPTS_FIELD_NAME}
            label={labelWithHint(
              t("adminPanel.loggingSettings.fields.storePrompts.label"),
              describeField(STORE_PROMPTS_FIELD_NAME, t("adminPanel.loggingSettings.fields.storePrompts.hint")),
            )}
          >
            {({ id, value, onChange, onBlur }) => (
              <Switch id={id} checked={Boolean(value)} onCheckedChange={onChange} onBlur={onBlur} className="w-fit" />
            )}
          </FormField>

          {OPTIONAL_FIELDS.map((field) => (
            <FormField
              key={field.name}
              control={form.control}
              name={field.name}
              label={labelWithHint(t(field.label), describeField(field.name, t(field.fallbackTooltip)))}
            >
              {({ ref, onChange, onBlur, ...control }) =>
                field.kind === "duration" ? (
                  <InputGroup>
                    <InputGroupInput
                      {...control}
                      ref={ref}
                      onChange={(event) => onChange(event.target.value)}
                      onBlur={onBlur}
                      placeholder={t(field.placeholder)}
                    />
                    <InputGroupAddon>
                      <Clock />
                    </InputGroupAddon>
                  </InputGroup>
                ) : (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    onChange={(event) => onChange(event.target.value)}
                    onBlur={(event) => {
                      onChange(countDisplay(event.target.value));
                      onBlur();
                    }}
                    placeholder={t(field.placeholder)}
                  />
                )
              }
            </FormField>
          ))}

          <FormField
            control={form.control}
            name={SKIP_PASSED_FIELD_NAME}
            label={labelWithHint(
              t("adminPanel.loggingSettings.fields.guardrailLogSkipPassed.label"),
              describeField(SKIP_PASSED_FIELD_NAME, t("adminPanel.loggingSettings.fields.guardrailLogSkipPassed.hint")),
            )}
            description={t(
              skipPassedEditable
                ? "adminPanel.loggingSettings.fields.guardrailLogSkipPassed.description"
                : "adminPanel.loggingSettings.fields.guardrailLogSkipPassed.configOwned",
            )}
          >
            {({ id, value, onChange, onBlur }) => (
              <Switch
                id={id}
                disabled={!skipPassedEditable}
                checked={Boolean(value)}
                onCheckedChange={onChange}
                onBlur={onBlur}
                className="w-fit"
              />
            )}
          </FormField>
        </FieldGroup>

        <Button type="submit" className="mt-6" disabled={isSaving}>
          {isSaving && (
            <UiLoadingSpinner
              role="img"
              aria-label={t("adminPanel.loggingSettings.savingAriaLabel")}
              className="size-4"
            />
          )}
          {isSaving ? t("adminPanel.loggingSettings.saving") : t("adminPanel.loggingSettings.save")}
        </Button>
      </form>
    </TooltipProvider>
  );
};

const LoggingSettings: React.FC = () => {
  const { t } = useTranslation();
  const { mutate, isPending } = useStoreRequestInSpendLogs();
  const { mutate: deleteField, isPending: isDeletingField } = useDeleteProxyConfigField();
  const { data: proxyConfigData, isLoading: isLoadingConfig } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  const describeField = (name: string, fallback: string) =>
    proxyConfigData?.find((field) => field.field_name === name)?.field_description || fallback;

  const storedValue = useCallback(
    (name: string) => proxyConfigData?.find((field) => field.field_name === name)?.field_value,
    [proxyConfigData],
  );

  const skipPassedEditable =
    proxyConfigData?.find((field) => field.field_name === SKIP_PASSED_FIELD_NAME)?.editable !== false;

  const isStored = (name: GeneralSettingsFieldName) => {
    const value = storedValue(name);
    return value !== null && value !== undefined;
  };

  const initialValues = useMemo(() => storedValuesToForm(storedValue), [storedValue]);

  // Resolves to the field name when clearing it failed, or null when it worked.
  const clearStoredField = (fieldName: GeneralSettingsFieldName) =>
    new Promise<GeneralSettingsFieldName | null>((resolve) => {
      let failed = false;
      deleteField(
        { config_type: ConfigType.GENERAL_SETTINGS, field_name: fieldName },
        {
          onError: () => {
            failed = true;
          },
          onSettled: () => resolve(failed ? fieldName : null),
        },
      );
    });

  // Clearing a field rewrites the whole stored general_settings object server
  // side, so these must run one at a time: in parallel the last write back wins
  // and silently restores the fields the earlier ones just cleared.
  const clearStoredFieldsInSequence = async (
    fieldNames: readonly GeneralSettingsFieldName[],
  ): Promise<readonly GeneralSettingsFieldName[]> => {
    const failed: GeneralSettingsFieldName[] = [];
    for (const fieldName of fieldNames) {
      const failure = await clearStoredField(fieldName);
      if (failure !== null) {
        failed.push(failure);
      }
    }
    return failed;
  };

  const handleFormSubmit = (formValues: LoggingSettingsFormValues) => {
    // A value the config file owns is rejected by the proxy when it changes, so it is only sent when editable.
    const saveSkipPassed =
      skipPassedEditable && formValues.guardrail_log_skip_passed !== initialValues.guardrail_log_skip_passed;
    const updateParams = formToPayload(formValues, saveSkipPassed);
    const submitUpdate = () =>
      mutate(updateParams, {
        onSuccess: () => toast.success(t("adminPanel.loggingSettings.toasts.updated")),
        onError: (error) =>
          toast.fromError(t("adminPanel.loggingSettings.toasts.saveFailed", { error: parseErrorMessage(error) })),
      });

    const fieldsToClear = omittedFieldNames(
      updateParams,
      OPTIONAL_FIELDS.map((field) => field.name),
      isStored,
    );
    if (fieldsToClear.length === 0) {
      submitUpdate();
      return;
    }

    void clearStoredFieldsInSequence(fieldsToClear).then((failed) => {
      if (failed.length > 0) {
        // Reporting an unqualified success here would tell the admin a setting
        // was reset to its default while the old value is still in force.
        toast.fromError(t("adminPanel.loggingSettings.toasts.clearFailed", { fields: failed.join(", ") }));
        return;
      }
      submitUpdate();
    });
  };

  return (
    <Card>
      <CardHeader className="border-b">
        <CardTitle>{t("adminPanel.loggingSettings.title")}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex w-full flex-col gap-6">
          <p className="mb-0 text-muted-foreground">{t("adminPanel.loggingSettings.description")}</p>

          {isLoadingConfig ? (
            <div className="flex flex-col gap-3">
              <Skeleton className="h-4 w-2/5" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/5" />
            </div>
          ) : (
            <LoggingSettingsForm
              initialValues={initialValues}
              describeField={describeField}
              skipPassedEditable={skipPassedEditable}
              isSaving={isPending || isDeletingField}
              onSubmit={handleFormSubmit}
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default LoggingSettings;
