import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import type { FormInstance } from "antd";
import { Form } from "antd";
import type { UploadProps } from "antd/es/upload";
import React from "react";
import type { Team } from "../key_team_helpers/key_list";
import { type CredentialItem } from "../networking";
import { Providers } from "../provider_info_helpers";
import AddAutoRouterTab from "./add_auto_router_tab";
import AddModelForm from "./AddModelForm";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";

interface AddModelTabProps {
  form: FormInstance;
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
  const [autoRouterForm] = Form.useForm();

  const handleAutoRouterOk = () => {
    autoRouterForm
      .validateFields()
      .then((values) => {
        handleAddAutoRouterSubmit(
          values,
          accessToken,
          autoRouterForm,
          handleOk,
        );
      })
      .catch((error) => {
        console.error("Validation failed:", error);
      });
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
