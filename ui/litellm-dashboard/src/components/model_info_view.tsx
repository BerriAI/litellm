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

interface ModelInfoViewProps {
  modelId: string;
  onClose: () => void;
  modelData: any;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  editModel: boolean;
}

export default function ModelInfoView({ 
  modelId, 
  onClose, 
  modelData, 
  accessToken,
  userID,
  userRole,
  editModel
}: ModelInfoViewProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const canEditModel = userRole === "Admin";

  if (!modelData) {
    return (
      <div className="p-4">
        <Button 
          icon={ArrowLeftIcon} 
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

  const handleModelUpdate = async (values: any) => {
    try {
      if (!accessToken) return;
      
      let payload = {
        litellm_params: Object.keys(values.litellm_params).length > 0 ? values.litellm_params : undefined,
        model_info: values.model_info ? {
          id: values.model_info.id,
        } : undefined,
      };

      await modelUpdateCall(accessToken, payload);
      message.success("Model updated successfully");
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating model:", error);
      message.error("Failed to update model");
    }
  };

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">← Back</Button>
          <Title>{modelData.model_name}</Title>
          <Text className="text-gray-500 font-mono">{modelData.model_info.id}</Text>
        </div>
        {canEditModel && (
          <div className="flex gap-2">
            <TremorButton
              onClick={() => setIsDeleteModalOpen(true)}
              color="red"
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
                <div className="mt-2">
                  <Title>{modelData.provider || "Not Set"}</Title>
                </div>
              </Card>

              <Card>
                <Text>Pricing</Text>
                <div className="mt-2">
                  <Text>Input: ${modelData.input_cost}/1M tokens</Text>
                  <Text>Output: ${modelData.output_cost}/1M tokens</Text>
                </div>
              </Card>

              <Card>
                <Text>Token Limits</Text>
                <div className="mt-2">
                  <Text>Max Tokens: {modelData.max_tokens || "Not Set"}</Text>
                  <Text>Max Input Tokens: {modelData.max_input_tokens || "Not Set"}</Text>
                </div>
              </Card>
            </Grid>

            {/* Settings Card */}
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Model Settings</Title>
                {(canEditModel && !isEditing) && (
                  <TremorButton 
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
                  onSubmit={handleModelUpdate}
                />
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Model ID</Text>
                    <div className="font-mono">{modelData.model_info.id}</div>
                  </div>
                  
                  <div>
                    <Text className="font-medium">Public Model Name</Text>
                    <div>{modelData.model_name}</div>
                  </div>

                  <div>
                    <Text className="font-medium">LiteLLM Model Name</Text>
                    <div>{modelData.litellm_model_name}</div>
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

                  {userRole === "Admin" && (
                    <div>
                      <Text className="font-medium">API Base</Text>
                      <div>{modelData.api_base || "Not Set"}</div>
                    </div>
                  )}

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