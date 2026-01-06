import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import type { FormInstance } from "antd";
import { Form } from "antd";
import { PlusCircleOutlined } from "@ant-design/icons";
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
      <div className="mb-6 px-5 py-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100 flex items-center gap-5">
        <div className="flex-shrink-0 w-11 h-11 bg-white rounded-full flex items-center justify-center shadow-sm border border-blue-100">
          <PlusCircleOutlined style={{ fontSize: '22px', color: '#4F46E5' }} />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-gray-900 font-semibold text-[15px] m-0 leading-tight">Missing a provider?</h4>
          <p className="text-gray-500 text-[13px] m-0 mt-0.5 leading-snug">
            We're constantly adding support for new LLM models, providers, endpoints. If you don't see the one you need, let us know and we'll prioritize it.
          </p>
        </div>
        <a
          href="https://github.com/BerriAI/litellm/issues/18686"
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg transition-colors shadow-sm"
        >
          Request Provider
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </a>
      </div>
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
