"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
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
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const INITIAL_VALUES: MountedFormValues = { litellm_credential_name: null };

export default function AddModelPanel() {
  const { accessToken } = useAuthorized();
  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues: INITIAL_VALUES });
  const registry = useMountRegistry();
  const queryClient = useQueryClient();
  const { data: modelCostMapData } = useModelCostMap();
  const { data: credentialsResponse } = useCredentials();
  const { data: teams } = useTeams();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.Anthropic);
  const [providerModels, setProviderModels] = useState<string[]>([]);
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
      { resetFields: () => form.reset(INITIAL_VALUES) },
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
      setProviderModelsFn={(provider) => setProviderModels(getProviderModels(provider, modelCostMapData))}
      getPlaceholder={getPlaceholder}
      showAdvancedSettings={showAdvancedSettings}
      setShowAdvancedSettings={setShowAdvancedSettings}
      teams={teams ?? null}
      credentials={credentialsResponse?.credentials || []}
    />
  );
}
