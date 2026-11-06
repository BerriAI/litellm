"use client";

import { useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { modelCreateCall, modelPatchUpdateCall } from "@/components/networking";
import {
  emptySwitchyardConfig,
  isDecisionModelMode,
  switchyardConfigSchema,
  type SwitchyardConfig,
} from "./switchyard_config";
import SwitchyardConfigFields from "./SwitchyardConfigFields";

interface Props {
  hidden?: boolean;
  accessToken: string;
  onSuccess: (saved: { model_name: string; switchyard_router_config: SwitchyardConfig }) => void;
  routerName?: string;
  onRouterNameChange?: (name: string) => void;
  initial?: { name: string; modelId: string; config: unknown };
}

export default function SwitchyardRouterForm({
  accessToken,
  onSuccess,
  initial,
  routerName,
  onRouterNameChange,
  hidden = false,
}: Props) {
  const id = useId();
  const parsed = initial ? switchyardConfigSchema.safeParse(initial.config) : undefined;
  const [config, setConfig] = useState<SwitchyardConfig>(parsed?.success ? parsed.data : emptySwitchyardConfig());
  const [localName, setLocalName] = useState(initial?.name ?? "");
  const name = routerName ?? localName;
  const setName = onRouterNameChange ?? setLocalName;
  const [error, setError] = useState<string | null>(
    parsed && !parsed.success ? "The stored Switchyard configuration is invalid. Correct it before saving." : null,
  );
  const [toolError, setToolError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const { data: models = [], isError } = useQuery({
    queryKey: ["availableModels", "switchyard", accessToken],
    queryFn: () => fetchAvailableModels(accessToken),
    enabled: Boolean(accessToken) && !hidden,
  });
  const routable = models.filter(
    (model) => model.model_group !== name && !model.model_group.startsWith("auto_router/"),
  );
  const toChoice = (model: { model_group: string }) => ({ value: model.model_group, label: model.model_group });
  const choices = routable.filter((model) => !isDecisionModelMode(model.mode)).map(toChoice);
  const decisionChoices = routable.filter((model) => isDecisionModelMode(model.mode)).map(toChoice);
  const save = async () => {
    const validated = switchyardConfigSchema.safeParse(config);
    if (!name.trim()) {
      setError("Enter a router name");
      return;
    }
    if (!validated.success) {
      setError(validated.error.issues[0].message);
      return;
    }
    if (Object.values(validated.data.models).includes(name.trim())) {
      setError("The router cannot select itself");
      return;
    }
    if (toolError) return;
    setSaving(true);
    setError(null);
    try {
      const payload = {
        model_name: name.trim(),
        litellm_params: { model: "auto_router/switchyard", switchyard_router_config: validated.data },
      };
      if (initial) await modelPatchUpdateCall(accessToken, payload, initial.modelId);
      else await modelCreateCall(accessToken, { ...payload, model_info: {} });
      onSuccess({ model_name: payload.model_name, switchyard_router_config: validated.data });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Unable to save Switchyard router");
    } finally {
      setSaving(false);
    }
  };
  const submitLabel = initial ? "Save changes" : "Create router";
  return (
    <form
      hidden={hidden}
      className="space-y-6"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <div className="space-y-2">
        <label htmlFor={`${id}-name`} className="text-sm font-medium">
          Router name
        </label>
        <Input
          id={`${id}-name`}
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="stage-router"
        />
      </div>
      <p className="text-sm text-muted-foreground">
        Route between a capable and an efficient model using the conversation’s tool history.
      </p>
      {isError && (
        <p role="alert" className="text-sm text-destructive">
          Unable to load available models.
        </p>
      )}
      <SwitchyardConfigFields
        value={config}
        onChange={setConfig}
        models={choices}
        decisionModels={decisionChoices}
        onToolSemanticsError={setToolError}
      />
      {(error || toolError) && (
        <p role="alert" className="text-sm text-destructive">
          {toolError || error}
        </p>
      )}
      <div className="flex justify-end">
        <Button type="submit" disabled={saving || Boolean(toolError)}>
          {saving ? "Saving…" : submitLabel}
        </Button>
      </div>
    </form>
  );
}
