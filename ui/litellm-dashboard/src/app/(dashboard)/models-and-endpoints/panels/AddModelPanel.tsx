"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useForm, useWatch } from "react-hook-form";
import { parseAsString, useQueryStates } from "nuqs";
import { useQueryClient } from "@tanstack/react-query";
import AddModelForm from "@/components/add_model/AddModelForm";
import { handleAddModelSubmit } from "@/components/add_model/handle_add_model_submit";
import {
  projectMountedValues,
  useMountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import { Providers, getPlaceholder, getProviderModels } from "@/components/provider_info_helpers";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { buildDecisionCatalog } from "@/lib/decisionModels";

const INITIAL_VALUES: MountedFormValues = { litellm_credential_name: null };

const PREFILL_URL_PARAMS = { provider: parseAsString, litellm_model: parseAsString };

const prefillMapping = (provider: string, model: string) => ({
  public_name: model,
  litellm_model: provider === Providers.Azure ? `azure/${model}` : model,
});

export default function AddModelPanel() {
  const { accessToken } = useAuthorized();
  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues: INITIAL_VALUES });
  const registry = useMountRegistry();
  const queryClient = useQueryClient();
  const { data: modelCostMapData } = useModelCostMap(true, true);
  const { data: credentialsResponse } = useCredentials();
  const { data: teams } = useTeams();
  const [selectedProvider, setSelectedProvider] = useState<string | null>(Providers.Anthropic);
  const [{ provider: prefillProviderSlug, litellm_model: prefillModel }, setPrefillParams] =
    useQueryStates(PREFILL_URL_PARAMS);
  const { data: providerMetadata } = useProviderFields();
  const prefillProvider = useMemo(
    () =>
      prefillProviderSlug === null
        ? null
        : providerMetadata?.find((info) => info.litellm_provider === prefillProviderSlug)?.provider ?? null,
    [prefillProviderSlug, providerMetadata],
  );
  const prefillApplied = useRef(false);

  useEffect(() => {
    if (prefillApplied.current || prefillProvider === null) {
      return;
    }
    prefillApplied.current = true;
    setSelectedProvider(prefillProvider);
    form.setValue("custom_llm_provider", prefillProvider);
    if (prefillModel === null) {
      return;
    }
    form.setValue("model", [prefillModel]);
    form.setValue("model_mappings", [prefillMapping(prefillProvider, prefillModel)]);
  }, [form, prefillModel, prefillProvider]);

  const pickedProvider = useWatch({ control: form.control, name: "custom_llm_provider" });
  const providerModels = useMemo(
    () => (typeof pickedProvider === "string" ? getProviderModels(pickedProvider, modelCostMapData) : []),
    [pickedProvider, modelCostMapData],
  );
  const decisionCatalog = useMemo(() => buildDecisionCatalog(modelCostMapData), [modelCostMapData]);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  const refresh = () => queryClient.invalidateQueries({ queryKey: ["models", "list"] });

  const mountedValues = () => projectMountedValues(registry, form.getValues);

  const handleOk = async (): Promise<boolean> => {
    const isValid = await form.trigger(registry.mountedNames() as string[]);
    if (!isValid) {
      return false;
    }
    await handleAddModelSubmit(
      mountedValues(),
      accessToken,
      {
        resetFields: () => {
          form.reset(INITIAL_VALUES);
          void setPrefillParams({ provider: null, litellm_model: null });
        },
      },
      refresh,
    );
    return true;
  };

  return (
    <AddModelForm
      form={form}
      registry={registry}
      mountedValues={mountedValues}
      handleOk={handleOk}
      selectedProvider={selectedProvider}
      setSelectedProvider={setSelectedProvider}
      providerModels={providerModels}
      getPlaceholder={getPlaceholder}
      showAdvancedSettings={showAdvancedSettings}
      setShowAdvancedSettings={setShowAdvancedSettings}
      teams={teams ?? null}
      credentials={credentialsResponse?.credentials || []}
      decisionCatalog={decisionCatalog}
    />
  );
}
