import React, { useState } from "react";
import { Card, Text, Button, Grid, Col, TabGroup, TabList, Tab, TabPanels, TabPanel, Title, Badge } from "@tremor/react";
import { KeyResponse } from "./all_keys_table";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { Form, Input, InputNumber, message } from "antd";

interface KeyInfoViewProps {
  keyId: string;
  onClose: () => void;
  keyData: KeyResponse | undefined;
}

export default function KeyInfoView({ keyId, onClose, keyData }: KeyInfoViewProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();

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

  const handleKeyUpdate = async (values: any) => {
    try {
      // TODO: Implement key update API call
      message.success("Key updated successfully");
      setIsEditing(false);
    } catch (error) {
      message.error("Failed to update key");
      console.error("Error updating key:", error);
    }
  };

  return (
    <div className="p-4">
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
      </div>

      <TabGroup>
        <TabList className="mb-4">
          <Tab>Overview</Tab>
          <Tab>Settings</Tab>
          <Tab>Usage</Tab>
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
            </Grid>
          </TabPanel>

          {/* Settings Panel */}
          <TabPanel>
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Key Settings</Title>
                {!isEditing && (
                  <Button variant="light" onClick={() => setIsEditing(true)}>
                    Edit Settings
                  </Button>
                )}
              </div>

              {isEditing ? (
                <Form
                  form={form}
                  onFinish={handleKeyUpdate}
                  initialValues={keyData}
                  layout="vertical"
                >
                  <Form.Item label="Key Alias" name="key_alias">
                    <Input />
                  </Form.Item>

                  <Form.Item label="Max Budget (USD)" name="max_budget">
                    <InputNumber step={0.01} precision={2} style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item label="TPM Limit" name="tpm_limit">
                    <InputNumber style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item label="RPM Limit" name="rpm_limit">
                    <InputNumber style={{ width: "100%" }} />
                  </Form.Item>

                  <div className="flex justify-end gap-2 mt-6">
                    <Button variant="light" onClick={() => setIsEditing(false)}>
                      Cancel
                    </Button>
                    <Button type="primary" htmlType="submit">
                      Save Changes
                    </Button>
                  </div>
                </Form>
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
                  </div>

                  <div>
                    <Text className="font-medium">Metadata</Text>
                    <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">
                      {JSON.stringify(keyData.metadata, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </Card>
          </TabPanel>
          {/* Usage Panel */}
          <TabPanel>
            <Card>
              <Title>Usage Statistics</Title>
              {/* Add usage statistics here */}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
} 