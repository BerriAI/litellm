import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import React from "react";
import { useForm, UseFormReturn } from "react-hook-form";
import type { Team } from "../key_team_helpers/key_list";
import { type CredentialItem } from "../networking";
import { Providers } from "../provider_info_helpers";
import AddAutoRouterTab, {
  type AutoRouterFormValues,
} from "./add_auto_router_tab";
import AddModelForm, { type AddModelFormValues } from "./AddModelForm";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import type { UploadProps } from "./add_model_upload_types";

interface AddModelTabProps {
  form: UseFormReturn<AddModelFormValues>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handleOk: (values?: any) => Promise<void>;
  selectedProvider: Providers;
  setSelectedProvider: (provider: Providers) => void;
  providerModels: string[];
  setProviderModelsFn: (provider: Providers) => void;
  getPlaceholder: (provider: Providers) => string;
  uploadProps: UploadProps;
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams: Team[] | null;
  credentials: CredentialItem[];
  accessToken: string;
  userRole: string;
}

const AddModelTab: React.FC<AddModelTabProps> = ({
  form,
  handleOk,
  selectedProvider,
  setSelectedProvider,
  providerModels,
  setProviderModelsFn,
  getPlaceholder,
  uploadProps,
  showAdvancedSettings,
  setShowAdvancedSettings,
  teams,
  credentials,
  accessToken,
  userRole,
}) => {
  const autoRouterForm = useForm<AutoRouterFormValues>({
    defaultValues: {
      auto_router_name: "",
      auto_router_default_model: "",
      auto_router_embedding_model: "",
      model_access_group: [],
    },
  });

  const handleAutoRouterOk = async (values: AutoRouterFormValues) => {
    await handleAddAutoRouterSubmit(
      values,
      accessToken,
      autoRouterForm,
      () => handleOk(),
    );
  };

  return (
    <Tabs defaultValue="add-model" className="w-full">
      <TabsList className="mb-4">
        <TabsTrigger value="add-model">Add Model</TabsTrigger>
        <TabsTrigger value="add-auto-router">Add Auto Router</TabsTrigger>
      </TabsList>
      <TabsContent value="add-model">
        <AddModelForm
          form={form}
          handleOk={handleOk}
          selectedProvider={selectedProvider}
          setSelectedProvider={setSelectedProvider}
          providerModels={providerModels}
          setProviderModelsFn={setProviderModelsFn}
          getPlaceholder={getPlaceholder}
          uploadProps={uploadProps}
          showAdvancedSettings={showAdvancedSettings}
          setShowAdvancedSettings={setShowAdvancedSettings}
          teams={teams}
          credentials={credentials}
        />
      </TabsContent>
      <TabsContent value="add-auto-router">
        <AddAutoRouterTab
          form={autoRouterForm}
          handleOk={handleAutoRouterOk}
          accessToken={accessToken}
          userRole={userRole}
        />
      </TabsContent>
    </Tabs>
  );
};

export default AddModelTab;
