import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
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
  form: FormInstance; // For the Add Model tab
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
  // Create separate form instance for auto router
  const [autoRouterForm] = Form.useForm();

  const handleAutoRouterOk = () => {
    autoRouterForm
      .validateFields()
      .then((values) => {
        handleAddAutoRouterSubmit(values, accessToken, autoRouterForm, handleOk);
      })
      .catch((error) => {
        console.error("Validation failed:", error);
      });
  };

  return (
    <>
      <TabGroup className="w-full">
        <TabList className="mb-4">
          <Tab>Add Model</Tab>
          <Tab>Add Auto Router</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
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
          </TabPanel>
          <TabPanel>
            <AddAutoRouterTab
              form={autoRouterForm}
              handleOk={handleAutoRouterOk}
              accessToken={accessToken}
              userRole={userRole}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </>
  );
};

export default AddModelTab;
