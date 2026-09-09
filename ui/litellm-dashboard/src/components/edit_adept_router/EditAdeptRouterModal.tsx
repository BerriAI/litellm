"use client";

import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { fetchAvailableModels, type ModelGroup } from "@/components/llm_calls/fetch_models";
import { modelPatchUpdateCall } from "../networking";
import ModelChoiceCombobox from "../add_model/ModelChoiceCombobox";

const PG_SSL_MODES = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"] as const;

interface AdeptRouterLitellmParams {
  adept_router_default_model?: string | null;
  adept_router_tag_prefix?: string | null;
  adept_router_conversations_threshold?: number | null;
  adept_router_trainer_url?: string | null;
  adept_router_pg_host?: string | null;
  adept_router_pg_port?: number | null;
  adept_router_pg_database?: string | null;
  adept_router_pg_user?: string | null;
  adept_router_pg_password?: string | null;
  adept_router_pg_ssl_mode?: string | null;
}

export interface AdeptRouterModelData {
  model_name: string;
  litellm_params?: AdeptRouterLitellmParams;
  model_info: { id: string };
}

interface EditAdeptRouterModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: (updatedModel: AdeptRouterModelData) => void;
  modelData: AdeptRouterModelData;
  accessToken: string;
}

interface EditAdeptRouterFormValues {
  adept_router_name: string;
  adept_router_default_model: string;
  adept_router_tag_prefix: string;
  adept_router_conversations_threshold: number | null;
  adept_router_trainer_url: string;
  adept_router_pg_host: string;
  adept_router_pg_port: number | null;
  adept_router_pg_database: string;
  adept_router_pg_user: string;
  adept_router_pg_password: string;
  adept_router_pg_ssl_mode: string;
}

const toFormValues = (modelData: AdeptRouterModelData): EditAdeptRouterFormValues => {
  const lp = modelData.litellm_params ?? {};
  return {
    adept_router_name: modelData.model_name,
    adept_router_default_model: lp.adept_router_default_model ?? "",
    adept_router_tag_prefix: lp.adept_router_tag_prefix ?? "",
    adept_router_conversations_threshold: lp.adept_router_conversations_threshold ?? null,
    adept_router_trainer_url: lp.adept_router_trainer_url ?? "",
    adept_router_pg_host: lp.adept_router_pg_host ?? "",
    adept_router_pg_port: lp.adept_router_pg_port ?? 5432,
    adept_router_pg_database: lp.adept_router_pg_database ?? "",
    adept_router_pg_user: lp.adept_router_pg_user ?? "",
    adept_router_pg_password: "",
    adept_router_pg_ssl_mode: lp.adept_router_pg_ssl_mode ?? "prefer",
  };
};

const EditAdeptRouterModal: React.FC<EditAdeptRouterModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
}) => {
  const form = useForm<EditAdeptRouterFormValues>({ defaultValues: toFormValues(modelData) });
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [saving, setSaving] = useState(false);
  const pgPasswordAlreadySet = !!modelData.litellm_params?.adept_router_pg_password;

  useEffect(() => {
    if (isVisible) {
      form.reset(toFormValues(modelData));
    }
  }, [isVisible, modelData, form]);

  useEffect(() => {
    if (!isVisible) return;
    const loadModels = async () => {
      try {
        setModelInfo(await fetchAvailableModels(accessToken));
      } catch {
        // model list unavailable; combobox still accepts free-text
      }
    };
    loadModels();
  }, [isVisible, accessToken]);

  const modelChoices = Array.from(new Set(modelInfo.map((m) => m.model_group))).map((g) => ({ value: g, label: g }));

  const handleSave = async (values: EditAdeptRouterFormValues) => {
    setSaving(true);
    try {
      const updatedLitellmParams: AdeptRouterLitellmParams & { model: string } = {
        model: `adept/${values.adept_router_name}`,
        adept_router_default_model: values.adept_router_default_model || null,
        adept_router_tag_prefix: values.adept_router_tag_prefix || null,
        adept_router_conversations_threshold: values.adept_router_conversations_threshold ?? null,
        adept_router_trainer_url: values.adept_router_trainer_url || null,
        adept_router_pg_host: values.adept_router_pg_host || null,
        adept_router_pg_port: values.adept_router_pg_port ?? null,
        adept_router_pg_database: values.adept_router_pg_database || null,
        adept_router_pg_user: values.adept_router_pg_user || null,
        adept_router_pg_ssl_mode: values.adept_router_pg_ssl_mode || null,
      };
      // Only overwrite the password when the operator typed a new one; otherwise the
      // stored value is preserved by omission.
      if (values.adept_router_pg_password) {
        updatedLitellmParams.adept_router_pg_password = values.adept_router_pg_password;
      }

      const patchPayload = {
        model_name: values.adept_router_name,
        litellm_params: updatedLitellmParams,
      };
      await modelPatchUpdateCall(accessToken, patchPayload, modelData.model_info.id);

      toast.success(`Updated ADEPT Router: ${values.adept_router_name}`);
      const updatedModel: AdeptRouterModelData = {
        ...modelData,
        model_name: values.adept_router_name,
        litellm_params: {
          ...(modelData.litellm_params ?? {}),
          ...updatedLitellmParams,
        },
      };
      onSuccess(updatedModel);
      onCancel();
    } catch (error) {
      console.error("Failed to update ADEPT router:", error);
      toast.fromError("Failed to update ADEPT router: " + error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Edit ADEPT Router</DialogTitle>
          <DialogDescription>
            Update the default model, trainer, or Postgres connection. The password is only written on save when you
            enter a new value.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleSave)} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="adept_router_name"
              label="Router Name"
              description="Changing this rewrites the model group and the alias (adept/<name>)."
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} />}
            </FormField>

            <FormField control={form.control} name="adept_router_default_model" label="Default Model">
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

            <FormField control={form.control} name="adept_router_tag_prefix" label="XML Tag Prefix">
              {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} />}
            </FormField>

            <FormField
              control={form.control}
              name="adept_router_conversations_threshold"
              label="Conversations Threshold"
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

            <FormField control={form.control} name="adept_router_trainer_url" label="Trainer URL">
              {({ ref, ...field }) => <Input {...field} ref={ref} type="url" value={field.value ?? ""} />}
            </FormField>

            <FormField control={form.control} name="adept_router_pg_host" label="PostgreSQL Host">
              {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} />}
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
              {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} />}
            </FormField>

            <FormField control={form.control} name="adept_router_pg_user" label="PostgreSQL User">
              {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} />}
            </FormField>

            <FormField
              control={form.control}
              name="adept_router_pg_password"
              label="PostgreSQL Password"
              description={pgPasswordAlreadySet ? "Leave blank to keep the current password." : undefined}
            >
              {({ ref, ...field }) => (
                <Input
                  {...field}
                  ref={ref}
                  type="password"
                  value={field.value ?? ""}
                  placeholder={pgPasswordAlreadySet ? "Password stored" : ""}
                  autoComplete="new-password"
                />
              )}
            </FormField>

            <FormField control={form.control} name="adept_router_pg_ssl_mode" label="PostgreSQL SSL Mode">
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
          </FieldGroup>

          <DialogFooter>
            <Button variant="outline" type="button" onClick={onCancel} disabled={saving}>
              Cancel
            </Button>
            <Button type="submit" disabled={saving}>
              {saving ? "Saving..." : "Save"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default EditAdeptRouterModal;
