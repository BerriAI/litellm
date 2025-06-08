import React, { useState } from "react";
import {
  Card,
  Text,
  Button,
  Grid,
  Col,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Title,
  Badge,
  TextInput,
  Select as TremorSelect
} from "@tremor/react";
import { ArrowLeftIcon, TrashIcon, RefreshIcon } from "@heroicons/react/outline";
import { keyDeleteCall, keyUpdateCall } from "./networking";
import { KeyResponse } from "./key_team_helpers/key_list";
import { Form, Input, InputNumber, message, Select, Tooltip } from "antd";
import { KeyEditView } from "./key_edit_view";
import { RegenerateKeyModal } from "./regenerate_key_modal";
import { rolesWithWriteAccess } from '../utils/roles';
import ObjectPermissionsView from "./object_permissions_view";

interface KeyInfoViewProps {
  keyId: string;
  onClose: () => void;
  keyData: KeyResponse | undefined;
  onKeyDataUpdate?: (data: Partial<KeyResponse>) => void;
  onDelete?: () => void;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  teams: any[] | null;
  premiumUser: boolean;
}

export default function KeyInfoView({ keyId, onClose, keyData, accessToken, userID, userRole, teams, onKeyDataUpdate, onDelete, premiumUser }: KeyInfoViewProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isRegenerateModalOpen, setIsRegenerateModalOpen] = useState(false);

  if (!keyData) {
    return (
      <div className="p-4">
        <Button 
          icon={ArrowLeftIcon} 
          variant="light"
          onClick={onClose}
          className="mb-4"
        >
          Back to Keys
        </Button>
        <Text>Key not found</Text>
      </div>
    );
  }

  const handleKeyUpdate = async (formValues: Record<string, any>) => {
    try {
      if (!accessToken) return;

      const currentKey = formValues.token;
      formValues.key = currentKey;

      // Handle object_permission updates
      if (formValues.vector_stores !== undefined) {
        formValues.object_permission = {
          ...keyData.object_permission,
          vector_stores: formValues.vector_stores || []
        };
        // Remove vector_stores from the top level as it should be in object_permission
        delete formValues.vector_stores;
      }

      // Convert metadata back to an object if it exists and is a string
      if (formValues.metadata && typeof formValues.metadata === "string") {
        try {
          const parsedMetadata = JSON.parse(formValues.metadata);
          formValues.metadata = {
            ...parsedMetadata,
            ...(formValues.guardrails?.length > 0 ? { guardrails: formValues.guardrails } : {}),
          };
        } catch (error) {
          console.error("Error parsing metadata JSON:", error);
          message.error("Invalid metadata JSON");
          return;
        }
      } else {
        formValues.metadata = {
          ...(formValues.metadata || {}),
          ...(formValues.guardrails?.length > 0 ? { guardrails: formValues.guardrails } : {}),
        };
      }

      // Convert budget_duration to API format
      if (formValues.budget_duration) {
        const durationMap: Record<string, string> = {
          daily: "24h",
          weekly: "7d",
          monthly: "30d"
        };
        formValues.budget_duration = durationMap[formValues.budget_duration];
      }

      const newKeyValues = await keyUpdateCall(accessToken, formValues);
      if (onKeyDataUpdate) {
        onKeyDataUpdate(newKeyValues)
      }
      message.success("Key updated successfully");
      setIsEditing(false);
      // Refresh key data here if needed
    } catch (error) {
      message.error("Failed to update key");
      console.error("Error updating key:", error);
    }
  };

  const handleDelete = async () => {
    try {
      if (!accessToken) return;
      await keyDeleteCall(accessToken as string, keyData.token);
      message.success("Key deleted successfully");
      if (onDelete) {
        onDelete()
      }
      onClose();
    } catch (error) {
      console.error("Error deleting the key:", error);
      message.error("Failed to delete key");
    }
  };

  return (
    <div className="w-full h-screen p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button 
            icon={ArrowLeftIcon} 
            variant="light"
            onClick={onClose}
            className="mb-4"
          >
            Back to Keys
          </Button>
          <Title>{keyData.key_alias || "API Key"}</Title>
          <Text className="text-gray-500 font-mono">{keyData.token}</Text>
        </div>
        {userRole && rolesWithWriteAccess.includes(userRole) && (
          <div className="flex gap-2">
            <Tooltip title={!premiumUser ? "This is a LiteLLM Enterprise feature, and requires a valid key to use." : ""}>
              <span className="inline-block">
                <Button
                  icon={RefreshIcon}
                  variant="secondary"
                  onClick={() => setIsRegenerateModalOpen(true)}
                  className="flex items-center"
                  disabled={!premiumUser}
                >
                  Regenerate Key
                </Button>
              </span>
            </Tooltip>
            <Button
              icon={TrashIcon}
              variant="secondary"
              onClick={() => setIsDeleteModalOpen(true)}
              className="flex items-center"
            >
              Delete Key
            </Button>
          </div>
        )}
      </div>

      {/* Add RegenerateKeyModal */}
      <RegenerateKeyModal
        selectedToken={keyData}
        visible={isRegenerateModalOpen}
        onClose={() => setIsRegenerateModalOpen(false)}
        accessToken={accessToken}
        premiumUser={premiumUser}
      />

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
                      Delete Key
                    </h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Are you sure you want to delete this key?
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button
                  onClick={handleDelete}
                  color="red"
                  className="ml-2"
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

      <TabGroup>
        <TabList className="mb-4">
          <Tab>Overview</Tab>
          <Tab>Settings</Tab>
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Spend</Text>
                <div className="mt-2">
                  <Title>${Number(keyData.spend).toFixed(4)}</Title>
                  <Text>of {keyData.max_budget !== null ? `$${keyData.max_budget}` : "Unlimited"}</Text>
                </div>
              </Card>

              <Card>
                <Text>Rate Limits</Text>
                <div className="mt-2">
                  <Text>TPM: {keyData.tpm_limit !== null ? keyData.tpm_limit : "Unlimited"}</Text>
                  <Text>RPM: {keyData.rpm_limit !== null ? keyData.rpm_limit : "Unlimited"}</Text>
                </div>
              </Card>

              <Card>
                <Text>Models</Text>
                <div className="mt-2 flex flex-wrap gap-2">
                  {keyData.models && keyData.models.length > 0 ? (
                    keyData.models.map((model, index) => (
                      <Badge key={index} color="red">
                        {model}
                      </Badge>
                    ))
                  ) : (
                    <Text>No models specified</Text>
                  )}
                </div>
              </Card>

              <Card>
                <ObjectPermissionsView 
                  objectPermission={keyData.object_permission} 
                  variant="inline"
                  accessToken={accessToken}
                />
              </Card>
            </Grid>
          </TabPanel>

          {/* Settings Panel */}
          <TabPanel>
            <Card className="overflow-y-auto max-h-[65vh]">
              <div className="flex justify-between items-center mb-4">
                <Title>Key Settings</Title>
                {!isEditing && userRole && rolesWithWriteAccess.includes(userRole) && (
                  <Button variant="light" onClick={() => setIsEditing(true)}>
                    Edit Settings
                  </Button>
                )}
              </div>

              {isEditing ? (
                <KeyEditView
                  keyData={keyData}
                  onCancel={() => setIsEditing(false)}
                  onSubmit={handleKeyUpdate}
                  teams={teams}
                  accessToken={accessToken}
                  userID={userID}
                  userRole={userRole}
                />
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Key ID</Text>
                    <Text className="font-mono">{keyData.token}</Text>
                  </div>
                  
                  <div>
                    <Text className="font-medium">Key Alias</Text>
                    <Text>{keyData.key_alias || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Secret Key</Text>
                    <Text className="font-mono">{keyData.key_name}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Team ID</Text>
                    <Text>{keyData.team_id || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Organization</Text>
                    <Text>{keyData.organization_id || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Created</Text>
                    <Text>{new Date(keyData.created_at).toLocaleString()}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Expires</Text>
                    <Text>{keyData.expires ? new Date(keyData.expires).toLocaleString() : "Never"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Spend</Text>
                    <Text>${Number(keyData.spend).toFixed(4)} USD</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Budget</Text>
                    <Text>{keyData.max_budget !== null ? `$${keyData.max_budget} USD` : "Unlimited"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Models</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {keyData.models && keyData.models.length > 0 ? (
                        keyData.models.map((model, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-blue-100 rounded text-xs"
                          >
                            {model}
                          </span>
                        ))
                      ) : (
                        <Text>No models specified</Text>
                      )}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Rate Limits</Text>
                    <Text>TPM: {keyData.tpm_limit !== null ? keyData.tpm_limit : "Unlimited"}</Text>
                    <Text>RPM: {keyData.rpm_limit !== null ? keyData.rpm_limit : "Unlimited"}</Text>
                    <Text>Max Parallel Requests: {keyData.max_parallel_requests !== null ? keyData.max_parallel_requests : "Unlimited"}</Text>
                    <Text>Model TPM Limits: {keyData.metadata?.model_tpm_limit ? JSON.stringify(keyData.metadata.model_tpm_limit) : "Unlimited"}</Text>
                    <Text>Model RPM Limits: {keyData.metadata?.model_rpm_limit ? JSON.stringify(keyData.metadata.model_rpm_limit) : "Unlimited"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Metadata</Text>
                    <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">
                      {JSON.stringify(keyData.metadata, null, 2)}
                    </pre>
                  </div>

                  <ObjectPermissionsView 
                    objectPermission={keyData.object_permission} 
                    variant="inline"
                    className="pt-4 border-t border-gray-200"
                    accessToken={accessToken}
                  />
                </div>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
} 