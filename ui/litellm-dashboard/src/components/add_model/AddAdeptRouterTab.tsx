"use client";

import React, { useEffect, useState } from "react";
import { useForm, useWatch, type UseFormReturn } from "react-hook-form";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { fetchAvailableModels, type ModelGroup } from "@/components/llm_calls/fetch_models";
import { modelAvailableCall } from "../networking";
import AccessGroupTagsCombobox from "./AccessGroupTagsCombobox";
import ModelChoiceCombobox from "./ModelChoiceCombobox";
import TeamDropdown from "../common_components/team_dropdown";
import { type ModelWriteScope } from "@/utils/modelPermissions";
import { handleAddAdeptRouterSubmit, type AddAdeptRouterValues } from "./HandleAddAdeptRouterSubmit";

// Fixed sslmode options mirror libpq's set. Kept in sync with the backend `LiteLLM_Params.adept_router_pg_ssl_mode`.
const PG_SSL_MODES = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"] as const;

const EMPTY_FORM_VALUES: AddAdeptRouterValues = {
  adept_router_name: "",
  adept_router_default_model: "",
  adept_router_tag_prefix: "",
  adept_router_conversations_threshold: null,
  adept_router_trainer_url: "",
  adept_router_pg_host: "",
  adept_router_pg_port: 5432,
  adept_router_pg_database: "",
  adept_router_pg_user: "",
  adept_router_pg_password: "",
  adept_router_pg_ssl_mode: "prefer",
  team_id: "",
  model_access_group: [],
};

interface AddAdeptRouterTabProps {
  handleOk?: () => void;
  accessToken: string;
  userRole: string;
  userId?: string | null;
  createScope?: ModelWriteScope;
}

const AddAdeptRouterTab: React.FC<AddAdeptRouterTabProps> = ({
  handleOk,
  accessToken,
  userRole,
  createScope = "unscoped-ok",
}) => {
  const requiresTeamScope = createScope === "team-required";
  const form: UseFormReturn<AddAdeptRouterValues> = useForm<AddAdeptRouterValues>({
    defaultValues: EMPTY_FORM_VALUES,
  });
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);

  const watchedName = useWatch({ control: form.control, name: "adept_router_name" });
  const watchedDefaultModel = useWatch({ control: form.control, name: "adept_router_default_model" });
  const watchedTeamId = useWatch({ control: form.control, name: "team_id" });

  useEffect(() => {
    const loadAccessGroups = async () => {
      try {
        const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
        setModelAccessGroups(response["data"].map((model: { id: string }) => model["id"]));
      } catch {
        // access groups unavailable; the combobox falls back to a plain text entry
      }
    };
    loadAccessGroups();
  }, [accessToken]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        setModelInfo(await fetchAvailableModels(accessToken));
      } catch {
        // model list unavailable; the combobox still renders a free-text entry
      }
    };
    loadModels();
  }, [accessToken]);

  const modelChoices = Array.from(new Set(modelInfo.map((m) => m.model_group))).map((g) => ({ value: g, label: g }));

  const computeSubmitBlockedReason = (): string | null => {
    if (!watchedName?.trim()) return "Enter a router name";
    if (!watchedDefaultModel?.trim()) return "Select a default fallback model";
    if (requiresTeamScope && !watchedTeamId?.trim()) return "Select a team to create this router under";
    return null;
  };
  const submitBlockedReason: string | null = computeSubmitBlockedReason();

  const onSubmit = async (values: AddAdeptRouterValues) => {
    if (submitBlockedReason !== null) {
      toast.fromError(submitBlockedReason);
      return;
    }
    await handleAddAdeptRouterSubmit(values, accessToken, () => form.reset(EMPTY_FORM_VALUES), handleOk);
  };

  return (
    <Card className="block p-6">
      <div className="mb-4">
        <h3 className="text-lg font-medium">Add ADEPT Router</h3>
        <p className="text-sm text-muted-foreground">
          Route XML-tagged agent prompts to task-specific SLMs. Requests fall back to the default model until a
          template&apos;s <code>target_model</code> is trained.
        </p>
      </div>
      <form onSubmit={form.handleSubmit(onSubmit)} noValidate>
        <FieldGroup>
          <FormField
            control={form.control}
            name="adept_router_name"
            label="Router Name"
            description="Model group name; the underlying alias is adept/<name>."
          >
            {({ ref, ...field }) => (
              <Input {...field} ref={ref} placeholder="e.g., adept_router_prod" value={field.value ?? ""} />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="adept_router_default_model"
            label="Default Model"
            description="Model that serves requests until a template is trained."
          >
            {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
              <ModelChoiceCombobox
                id={id}
                value={value ?? ""}
                onChange={onChange}
                choices={modelChoices}
                placeholder="Pick a fallback model"
                ariaInvalid={ariaInvalid}
                ariaDescribedBy={ariaDescribedBy}
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="adept_router_tag_prefix"
            label="XML Tag Prefix"
            description="Optional prefix for the XML tags the router masks (e.g. 'var' matches <var invoice_id>)."
          >
            {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="var" value={field.value ?? ""} />}
          </FormField>

          <FormField
            control={form.control}
            name="adept_router_conversations_threshold"
            label="Conversations Threshold"
            description="Trigger the trainer at every multiple of this count (default: 1000)."
          >
            {({ ref, value, onChange, ...field }) => (
              <Input
                {...field}
                ref={ref}
                type="number"
                min={1}
                value={value ?? ""}
                onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="adept_router_trainer_url"
            label="Trainer URL"
            description="External training pipeline webhook. Must be http:// or https:// and pass any allowed-hosts filter."
          >
            {({ ref, ...field }) => (
              <Input
                {...field}
                ref={ref}
                type="url"
                placeholder="https://trainer.internal/hooks"
                value={field.value ?? ""}
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="adept_router_pg_host"
            label="PostgreSQL Host"
            description="ADEPT stores templates + conversations in its own Postgres, separate from the proxy DB."
          >
            {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="db.internal" value={field.value ?? ""} />}
          </FormField>

          <FormField control={form.control} name="adept_router_pg_port" label="PostgreSQL Port">
            {({ ref, value, onChange, ...field }) => (
              <Input
                {...field}
                ref={ref}
                type="number"
                min={1}
                max={65535}
                value={value ?? ""}
                onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
              />
            )}
          </FormField>

          <FormField control={form.control} name="adept_router_pg_database" label="PostgreSQL Database">
            {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="adept" value={field.value ?? ""} />}
          </FormField>

          <FormField control={form.control} name="adept_router_pg_user" label="PostgreSQL User">
            {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="adept_rw" value={field.value ?? ""} />}
          </FormField>

          <FormField control={form.control} name="adept_router_pg_password" label="PostgreSQL Password">
            {({ ref, ...field }) => (
              <Input {...field} ref={ref} type="password" value={field.value ?? ""} autoComplete="new-password" />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="adept_router_pg_ssl_mode"
            label="PostgreSQL SSL Mode"
            description="libpq sslmode. Defaults to 'prefer'; use 'verify-full' when you have a CA configured."
          >
            {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
              <Select value={value ?? "prefer"} onValueChange={onChange}>
                <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
                  <SelectValue placeholder="prefer" />
                </SelectTrigger>
                <SelectContent>
                  {PG_SSL_MODES.map((mode) => (
                    <SelectItem key={mode} value={mode}>
                      {mode}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </FormField>

          {requiresTeamScope && (
            <FormField
              control={form.control}
              name="team_id"
              label="Team"
              description="Team admins must pick the team that owns this router."
            >
              {({ id, value, onChange }) => (
                <TeamDropdown id={id} value={value ?? ""} onChange={(next) => onChange(next ?? "")} />
              )}
            </FormField>
          )}

          <FormField
            control={form.control}
            name="model_access_group"
            label="Model Access Groups"
            description="Optional access groups virtual keys use to gate this router."
          >
            {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
              <AccessGroupTagsCombobox
                id={id}
                value={value ?? []}
                onChange={onChange}
                options={modelAccessGroups}
                ariaInvalid={ariaInvalid}
                ariaDescribedBy={ariaDescribedBy}
              />
            )}
          </FormField>
        </FieldGroup>

        <div className="mt-6 flex justify-end">
          <Button type="submit" disabled={submitBlockedReason !== null} title={submitBlockedReason ?? undefined}>
            {submitBlockedReason ?? "Create ADEPT Router"}
          </Button>
        </div>
      </form>
    </Card>
  );
};

export default AddAdeptRouterTab;
