import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Grid,
  Badge,
  Button as TremorButton,
} from "@tremor/react";
import { ArrowLeftIcon, TrashIcon } from "@heroicons/react/outline";
import { modelDeleteCall, modelUpdateCall } from "./networking";
import { Button, Form, Input, InputNumber, message, Select } from "antd";
import EditModelModal from "./edit_model/edit_model_modal";
import { handleEditModelSubmit } from "./edit_model/edit_model_modal";
import { getProviderLogoAndName } from "./provider_info_helpers";
import { getDisplayModelName } from "./view_model/model_name_display";

interface ModelInfoViewProps {
  modelId: string;
  onClose: () => void;
  modelData: any;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  editModel: boolean;
  setEditModalVisible: (visible: boolean) => void;
  setSelectedModel: (model: any) => void;
}

export default function ModelInfoView({ 
  modelId, 
  onClose, 
  modelData, 
  accessToken,
  userID,
  userRole,
  editModel,
  setEditModalVisible,
  setSelectedModel
}: ModelInfoViewProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const canEditModel = userRole === "Admin";

  if (!modelData) {
    return (
      <div className="p-4">
        <Button 
          icon={<ArrowLeftIcon />}
          onClick={onClose}
          className="mb-4"
        >
          Back to Models
        </Button>
        <Text>Model not found</Text>
      </div>
    );
  }

  const handleDelete = async () => {
    try {
      if (!accessToken) return;
      await modelDeleteCall(accessToken, modelId);
      message.success("Model deleted successfully");
      onClose();
    } catch (error) {
      console.error("Error deleting the model:", error);
      message.error("Failed to delete model");
    }
  };


  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button 
            icon={<ArrowLeftIcon />}
            onClick={onClose}
            className="mb-4"
          >
            Back to Models
          </Button>
          <Title>Public Model Name: {getDisplayModelName(modelData)}</Title>
          <Text className="text-gray-500 font-mono">{modelData.model_info.id}</Text>
        </div>
        {canEditModel && (
          <div className="flex gap-2">
            <TremorButton
              icon={TrashIcon}
              variant="secondary"
              onClick={() => setIsDeleteModalOpen(true)}
              className="flex items-center"
            >
              Delete Model
            </TremorButton>
          </div>
        )}
      </div>

      <TabGroup>
        <TabList className="mb-6">
          <Tab>Overview</Tab>
          <Tab>Raw JSON</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {/* Overview Grid */}
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6 mb-6">
              <Card>
                <Text>Provider</Text>
                <div className="mt-2 flex items-center space-x-2">
                  {modelData.provider && (
                    <img
                      src={getProviderLogoAndName(modelData.provider).logo}
                      alt={`${modelData.provider} logo`}
                      className="w-4 h-4"
                      onError={(e) => {
                        // Create a div with provider initial as fallback
                        const target = e.target as HTMLImageElement;
                        const parent = target.parentElement;
                        if (parent) {
                          const fallbackDiv = document.createElement('div');
                          fallbackDiv.className = 'w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs';
                          fallbackDiv.textContent = modelData.provider?.charAt(0) || '-';
                          parent.replaceChild(fallbackDiv, target);
                        }
                      }}
                    />
                  )}
                  <Title>{modelData.provider || "Not Set"}</Title>
                </div>
              </Card>
              <Card>
                <Text>LiteLLM Model</Text>
                <pre>
                  <Title>{modelData.litellm_model_name || "Not Set"}</Title>
                </pre>
              </Card>
              <Card>
                <Text>Pricing</Text>
                <div className="mt-2">
                  <Text>Input: ${modelData.input_cost}/1M tokens</Text>
                  <Text>Output: ${modelData.output_cost}/1M tokens</Text>
                </div>
              </Card>
            </Grid>

            {/* Settings Card */}
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Model Settings</Title>
                {(canEditModel && !isEditing) && (
                  <TremorButton 
                    variant="light"
                    onClick={() => setIsEditing(true)}
                  >
                    Edit Settings
                  </TremorButton>
                )}
              </div>

              {isEditing ? (
                <EditModelModal
                  visible={isEditing}
                  onCancel={() => setIsEditing(false)}
                  model={modelData}
                  onSubmit={(data: FormData) => handleEditModelSubmit(data, accessToken, setEditModalVisible, setSelectedModel)}
                />
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Model ID</Text>
                    <div className="font-mono">{modelData.model_info.id}</div>
                  </div>
                  
                  <div>
                    <Text className="font-medium">Public Model Name</Text>
                    <div>{getDisplayModelName(modelData)}</div>
                  </div>

                  <div>
                    <Text className="font-medium">LiteLLM Model Name</Text>
                    <div>{modelData.litellm_model_name}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Input Cost (per 1M tokens)</Text>
                    <div>{modelData.litellm_params?.input_cost_per_token ? (modelData.litellm_params.input_cost_per_token * 1_000_000).toFixed(4) : "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Output Cost (per 1M tokens)</Text>
                    <div>{modelData.litellm_params?.output_cost_per_token ? (modelData.litellm_params.output_cost_per_token * 1_000_000).toFixed(4) : "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">API Base</Text>
                    <div>{modelData.litellm_params?.api_base || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Custom LLM Provider</Text>
                    <div>{modelData.litellm_params?.custom_llm_provider || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Model</Text>
                    <div>{modelData.litellm_params?.model || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Organization</Text>
                    <div>{modelData.litellm_params?.organization || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">TPM (Tokens per Minute)</Text>
                    <div>{modelData.litellm_params?.tpm || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">RPM (Requests per Minute)</Text>
                    <div>{modelData.litellm_params?.rpm || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Max Retries</Text>
                    <div>{modelData.litellm_params?.max_retries || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Timeout (seconds)</Text>
                    <div>{modelData.litellm_params?.timeout || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Stream Timeout (seconds)</Text>
                    <div>{modelData.litellm_params?.stream_timeout || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Team ID</Text>
                    <div>{modelData.model_info.team_id || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Created At</Text>
                    <div>{modelData.model_info.created_at ? new Date(modelData.model_info.created_at).toLocaleString() : "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Created By</Text>
                    <div>{modelData.model_info.created_by || "Not Set"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">LiteLLM Parameters</Text>
                    <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">
                      {JSON.stringify(modelData.cleanedLitellmParams, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </Card>
          </TabPanel>

          <TabPanel>
            <Card>
              <pre className="bg-gray-100 p-4 rounded text-xs overflow-auto">
                {JSON.stringify(modelData, null, 2)}
              </pre>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Delete Model
                    </h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Are you sure you want to delete this model?
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button
                  onClick={handleDelete}
                  className="ml-2"
                  danger
                >
                  Delete
                </Button>
                <Button onClick={() => setIsDeleteModalOpen(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 