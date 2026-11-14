"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { CircleHelp } from "lucide-react";
import { z } from "zod";

import i18n from "@/i18n/i18n";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { createGuardrailCall, getGuardrailsList } from "@/components/networking";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import {
  buildCompressionGuardrailPayload,
  compressionGuardrailsOf,
  GuardrailListItem,
  GuardrailListResponse,
} from "./helpers";

interface PromptCompressionTabProps {
  accessToken: string | null;
}

const compressionSchema = z.object({
  name: z.string().min(1, { error: () => i18n.t("costOptimization.promptCompression.form.errors.nameRequired") }),
  apiBase: z.string().min(1, { error: () => i18n.t("costOptimization.promptCompression.form.errors.apiBaseRequired") }),
  defaultOn: z.boolean(),
});

type CompressionFormValues = z.infer<typeof compressionSchema>;

const EMPTY_VALUES: CompressionFormValues = {
  name: "",
  apiBase: "",
  defaultOn: true,
};

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const PromptCompressionTab: React.FC<PromptCompressionTabProps> = ({ accessToken }) => {
  const { t } = useTranslation();
  const form = useZodForm(compressionSchema, { defaultValues: EMPTY_VALUES });
  const [guardrails, setGuardrails] = useState<GuardrailListItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  const loadGuardrails = useCallback(() => {
    if (!accessToken) {
      return;
    }
    getGuardrailsList(accessToken)
      .then((response) => setGuardrails(compressionGuardrailsOf(response as GuardrailListResponse)))
      .catch((error) => {
        console.error("Failed to load compression guardrails:", error);
        toast.fromError(t("costOptimization.promptCompression.toasts.loadFailed"));
      })
      .finally(() => setIsLoading(false));
  }, [accessToken, t]);

  useEffect(() => {
    loadGuardrails();
  }, [loadGuardrails]);

  const handleAdd = async (values: CompressionFormValues) => {
    if (!accessToken) {
      return;
    }
    setIsSaving(true);
    try {
      await createGuardrailCall(
        accessToken,
        buildCompressionGuardrailPayload({
          name: values.name,
          apiBase: values.apiBase,
          defaultOn: values.defaultOn ?? true,
        }),
      );
      toast.success(t("costOptimization.promptCompression.toasts.created"));
      form.reset(EMPTY_VALUES);
      await loadGuardrails();
    } catch (error) {
      console.error("Failed to create compression guardrail:", error);
      toast.fromError(t("costOptimization.promptCompression.toasts.createFailed"));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>{t("costOptimization.promptCompression.overview.title")}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4 text-sm text-muted-foreground">
            <Trans
              i18nKey="costOptimization.promptCompression.overview.description"
              components={{
                docs: (
                  <a
                    href="https://docs.litellm.ai/docs/proxy/headroom"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-info underline"
                  />
                ),
              }}
            />
          </p>
          {isLoading && (
            <p className="text-sm text-muted-foreground">{t("costOptimization.promptCompression.overview.loading")}</p>
          )}
          {!isLoading && guardrails.length === 0 && (
            <p className="text-sm text-muted-foreground">{t("costOptimization.promptCompression.overview.empty")}</p>
          )}
          {!isLoading && guardrails.length > 0 && (
            <ul className="divide-y divide-border">
              {guardrails.map((guardrail) => (
                <li key={guardrail.guardrail_id} className="flex items-center justify-between py-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">{guardrail.guardrail_name}</p>
                    <p className="text-xs text-muted-foreground">{guardrail.litellm_params?.api_base ?? ""}</p>
                  </div>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      guardrail.litellm_params?.default_on
                        ? "bg-success/15 text-success"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {guardrail.litellm_params?.default_on
                      ? t("costOptimization.promptCompression.overview.alwaysOn")
                      : t("costOptimization.promptCompression.overview.optIn")}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t("costOptimization.promptCompression.form.title")}</CardTitle>
        </CardHeader>
        <CardContent>
          <TooltipProvider>
            <form onSubmit={form.handleSubmit(handleAdd)} noValidate>
              <FieldGroup>
                <FormField control={form.control} name="name" label={t("costOptimization.promptCompression.form.name")}>
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="headroom-compression" />}
                </FormField>
                <FormField
                  control={form.control}
                  name="apiBase"
                  label={labelWithHint(
                    t("costOptimization.promptCompression.form.apiBase.label"),
                    t("costOptimization.promptCompression.form.apiBase.hint"),
                  )}
                  description={t("costOptimization.promptCompression.form.apiBase.description")}
                >
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="https://your-headroom-endpoint" />}
                </FormField>
                <FormField
                  control={form.control}
                  name="defaultOn"
                  label={t("costOptimization.promptCompression.form.defaultOn")}
                >
                  {({ value, onChange, ref: _ref, ...field }) => (
                    <Switch
                      {...field}
                      nativeButton
                      render={<button type="button" />}
                      checked={value}
                      onCheckedChange={onChange}
                    />
                  )}
                </FormField>
              </FieldGroup>
              <div className="mt-6 mb-4 rounded-lg border border-warning/20 bg-warning/10 p-3">
                <p className="text-sm text-warning">
                  <Trans
                    i18nKey="costOptimization.promptCompression.form.enterpriseNotice"
                    components={{
                      pricing: (
                        <a
                          href="https://www.litellm.ai/#pricing"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline"
                        />
                      ),
                    }}
                  />
                </p>
              </div>
              <div className="flex justify-end">
                <Button type="submit" disabled={isSaving}>
                  {isSaving && <UiLoadingSpinner className="size-4" />}
                  {t("costOptimization.promptCompression.form.submit")}
                </Button>
              </div>
            </form>
          </TooltipProvider>
        </CardContent>
      </Card>
    </div>
  );
};

export default PromptCompressionTab;
