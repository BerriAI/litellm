"use client";

import React, { useCallback, useEffect, useState } from "react";
import { CircleHelp } from "lucide-react";
import { z } from "zod/v4";

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
  name: z.string().min(1, "Name is required"),
  apiBase: z.string().min(1, "API base is required"),
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
        toast.fromError("Failed to load compression guardrails");
      })
      .finally(() => setIsLoading(false));
  }, [accessToken]);

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
      toast.success("Compression guardrail created");
      form.reset(EMPTY_VALUES);
      await loadGuardrails();
    } catch (error) {
      console.error("Failed to create compression guardrail:", error);
      toast.fromError("Failed to create compression guardrail");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Headroom prompt compression</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4 text-sm text-muted-foreground">
            Headroom is a native LiteLLM guardrail that compresses your prompts before they reach the model, so you pay
            for fewer input tokens. The tokens it removes are priced and shown on the Usage tab as compression savings.{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/headroom"
              target="_blank"
              rel="noopener noreferrer"
              className="text-info underline"
            >
              Headroom setup docs
            </a>
          </p>
          {isLoading && <p className="text-sm text-muted-foreground">Loading...</p>}
          {!isLoading && guardrails.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No prompt compression guardrails configured yet. Add one below to start saving on input tokens
            </p>
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
                    {guardrail.litellm_params?.default_on ? "Always on" : "Opt-in"}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Add Headroom compression guardrail</CardTitle>
        </CardHeader>
        <CardContent>
          <TooltipProvider>
            <form onSubmit={form.handleSubmit(handleAdd)} noValidate>
              <FieldGroup>
                <FormField control={form.control} name="name" label="Name">
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="headroom-compression" />}
                </FormField>
                <FormField
                  control={form.control}
                  name="apiBase"
                  label={labelWithHint(
                    "Headroom API base",
                    "Base URL of your Headroom compression service (LiteLLM calls its /v1/compress endpoint)",
                  )}
                  description="The URL where your Headroom compression service is hosted"
                >
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="https://your-headroom-endpoint" />}
                </FormField>
                <FormField control={form.control} name="defaultOn" label="Apply to all requests">
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
                  Applying compression to all requests is available to all users. Enabling it selectively per key or
                  team is a LiteLLM Enterprise feature. Get a trial key{" "}
                  <a
                    href="https://www.litellm.ai/#pricing"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                  >
                    here
                  </a>
                </p>
              </div>
              <div className="flex justify-end">
                <Button type="submit" disabled={isSaving}>
                  {isSaving && <UiLoadingSpinner className="size-4" />}
                  Add guardrail
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
