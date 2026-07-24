"use client";

import { Form } from "antd";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import AddModelTab from "@/components/add_model/add_model_tab";
import { handleAddModelSubmit } from "@/components/add_model/handle_add_model_submit";
import { Providers, getPlaceholder, getProviderModels } from "@/components/provider_info_helpers";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { vertexCredentialsUploadProps } from "@/app/(dashboard)/models-and-endpoints/vertexCredentialsUpload";

export default function AddModelPage() {
  const { accessToken, userRole } = useAuthorized();
  const [form] = Form.useForm();
  const queryClient = useQueryClient();
  const { data: modelCostMapData } = useModelCostMap();
  const { data: credentialsResponse } = useCredentials();
  const { data: teams } = useTeams();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.Anthropic);
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  const refresh = () => queryClient.invalidateQueries({ queryKey: ["models", "list"] });

  const handleOk = async () => {
    try {
      const values = await form.validateFields();
      await handleAddModelSubmit(values, accessToken, form, refresh);
    } catch (error: any) {
      const errorMessages =
        error.errorFields?.map((field: any) => `${field.name.join(".")}: ${field.errors.join(", ")}`).join(" | ") ||
        "Unknown validation error";
      NotificationsManager.fromBackend(`Please fill in the following required fields: ${errorMessages}`);
    }
  };

  return (
    <AddModelTab
      form={form}
      handleOk={handleOk}
      selectedProvider={selectedProvider}
      setSelectedProvider={setSelectedProvider}
      providerModels={providerModels}
      setProviderModelsFn={(provider) => setProviderModels(getProviderModels(provider, modelCostMapData))}
      getPlaceholder={getPlaceholder}
      uploadProps={vertexCredentialsUploadProps(form)}
      showAdvancedSettings={showAdvancedSettings}
      setShowAdvancedSettings={setShowAdvancedSettings}
      teams={teams ?? null}
      credentials={credentialsResponse?.credentials || []}
      accessToken={accessToken}
      userRole={userRole}
    />
  );
}
