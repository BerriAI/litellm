"use client";

import { useId, useState } from "react";
import { useWatch } from "react-hook-form";
import { useZodForm } from "@/lib/forms/useZodForm";
import { useQuery } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { modelPatchUpdateCall } from "@/components/networking";
import { emptySwitchyardConfig, isDecisionModelMode, type SwitchyardConfig } from "./switchyard_config";
import SwitchyardConfigFields from "./SwitchyardConfigFields";
import { handleAddSwitchyardSubmit, type SwitchyardModelInfo } from "./handle_add_auto_router_submit";
import { useTranslation } from "react-i18next";
import { switchyardEditDefaults, toSwitchyardEditPayload } from "./switchyard-edit/mapper";
import { switchyardEditSchema } from "./switchyard-edit/schema";

interface FieldsProps {
  accessToken: string;
  routerName: string;
  value: SwitchyardConfig;
  onChange: (value: SwitchyardConfig) => void;
  onToolSemanticsError: (error: string | null) => void;
}

export function SwitchyardModelFields({ accessToken, routerName, value, onChange, onToolSemanticsError }: FieldsProps) {
  const { t } = useTranslation();
  const { data: models = [], isError } = useQuery({
    queryKey: ["availableModels", "switchyard", accessToken],
    queryFn: () => fetchAvailableModels(accessToken),
    enabled: Boolean(accessToken),
  });
  const routable = models.filter(
    (model) => model.model_group !== routerName && !model.model_group.startsWith("auto_router/"),
  );
  const toChoice = (model: { model_group: string }) => ({ value: model.model_group, label: model.model_group });
  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">{t("addModel.switchyardRouterForm.modelFields.description")}</p>
      {isError && (
        <p role="alert" className="text-sm text-destructive">
          {t("addModel.switchyardRouterForm.modelFields.modelsLoadFailed")}
        </p>
      )}
      <SwitchyardConfigFields
        value={value}
        onChange={onChange}
        models={routable.filter((model) => !isDecisionModelMode(model.mode)).map(toChoice)}
        decisionModels={routable.filter((model) => isDecisionModelMode(model.mode)).map(toChoice)}
        onToolSemanticsError={onToolSemanticsError}
      />
    </div>
  );
}

interface StageSeed {
  capable?: string;
  efficient?: string;
}

export function useSwitchyardOption(accessToken: string, routerName: string, enabled: boolean) {
  const [selected, setSelected] = useState(false);
  const [config, setConfig] = useState(emptySwitchyardConfig);
  const [toolSemanticsError, setToolSemanticsError] = useState<string | null>(null);
  const select = (seed: StageSeed) => {
    setSelected(true);
    setToolSemanticsError(null);
    setConfig((current) => ({
      ...current,
      models: {
        ...current.models,
        capable: current.models.capable || seed.capable || "",
        efficient: current.models.efficient || seed.efficient || "",
      },
    }));
  };
  return {
    selected,
    deselect: () => setSelected(false),
    option: (seed: StageSeed) =>
      enabled
        ? {
            selected,
            onSelect: () => select(seed),
            fields: (
              <SwitchyardModelFields
                accessToken={accessToken}
                routerName={routerName}
                value={config}
                onChange={setConfig}
                onToolSemanticsError={setToolSemanticsError}
              />
            ),
          }
        : undefined,
    submit: (name: string, modelInfo: SwitchyardModelInfo, resetForm: () => void, onSuccess: () => void) => {
      const draft = { name, config, toolSemanticsError, modelInfo };
      return handleAddSwitchyardSubmit(draft, accessToken, resetForm, onSuccess);
    },
  };
}

interface Props {
  accessToken: string;
  onSuccess: (saved: { model_name: string; switchyard_router_config: SwitchyardConfig }) => void;
  initial: { name: string; modelId: string; config: unknown };
}

export default function SwitchyardRouterForm({ accessToken, onSuccess, initial }: Props) {
  const { t } = useTranslation();
  const id = useId();
  const [initialDraft] = useState(() => switchyardEditDefaults(initial));
  const form = useZodForm(switchyardEditSchema, { defaultValues: initialDraft.values });
  const [name, config] = useWatch({ control: form.control, name: ["name", "config"] });
  const [error, setError] = useState(initialDraft.error);
  const [toolError, setToolError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const save = async () => {
    const result = switchyardEditSchema.safeParse(form.getValues());
    if (!result.success) {
      setError(result.error.issues[0].message);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const payload = toSwitchyardEditPayload(result.data);
      await modelPatchUpdateCall(accessToken, payload, initial.modelId);
      onSuccess({
        model_name: payload.model_name,
        switchyard_router_config: payload.litellm_params.switchyard_router_config,
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Unable to save Switchyard router");
    } finally {
      setSaving(false);
    }
  };
  return (
    <form
      className="space-y-6"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <div className="space-y-2">
        <label htmlFor={`${id}-name`} className="text-sm font-medium">
          {t("addModel.switchyardRouterForm.routerName")}
        </label>
        <Input id={`${id}-name`} value={name} onChange={(event) => form.setValue("name", event.target.value)} />
      </div>
      <SwitchyardModelFields
        accessToken={accessToken}
        routerName={name}
        value={config}
        onChange={(value) => form.setValue("config", value)}
        onToolSemanticsError={setToolError}
      />
      {(error || toolError) && (
        <p role="alert" className="text-sm text-destructive">
          {toolError || error}
        </p>
      )}
      <div className="flex justify-end">
        <Button type="submit" disabled={saving || Boolean(toolError)}>
          {saving ? t("addModel.switchyardRouterForm.saving") : t("addModel.switchyardRouterForm.saveChanges")}
        </Button>
      </div>
    </form>
  );
}
