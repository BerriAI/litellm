import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Grid,
  Badge,
  Button as TremorButton,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
} from "@tremor/react";
import { Button, Form, Input, Select, message, Tooltip } from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { getGuardrailInfo } from "@/components/networking";
import { getGuardrailLogoAndName } from "./guardrail_info_helpers";

export interface GuardrailInfoProps {
  guardrailId: string;
  onClose: () => void;
  accessToken: string | null;
  isAdmin: boolean;
}

const GuardrailInfoView: React.FC<GuardrailInfoProps> = ({ 
  guardrailId, 
  onClose, 
  accessToken,
  isAdmin
}) => {
  const [guardrailData, setGuardrailData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();

  const fetchGuardrailInfo = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await getGuardrailInfo(accessToken, guardrailId);
      setGuardrailData(response);
    } catch (error) {
      message.error("Failed to load guardrail information");
      console.error("Error fetching guardrail info:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGuardrailInfo();
  }, [guardrailId, accessToken]);


  const handleGuardrailUpdate = async (values: any) => {
    try {
      // Not implemented yet - will be added in the future
      message.info("Guardrail update functionality coming soon");
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating guardrail:", error);
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!guardrailData) {
    return <div className="p-4">Guardrail not found</div>;
  }

  // Format date helper function
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Format the provider display name and logo
  const { logo, displayName } = getGuardrailLogoAndName(guardrailData.litellm_params?.guardrail || "");

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">← Back</Button>
          <Title>{guardrailData.guardrail_name || "Unnamed Guardrail"}</Title>
          <Text className="text-gray-500 font-mono">{guardrailData.guardrail_id}</Text>
        </div>
      </div>

      <TabGroup>
        <TabList className="mb-4">
          <Tab key="overview">Overview</Tab>
          {isAdmin ? <Tab key="settings">Settings</Tab> : <></>}
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Provider</Text>
                <div className="mt-2 flex items-center space-x-2">
                  {logo && (
                    <img 
                      src={logo} 
                      alt={`${displayName} logo`} 
                      className="w-6 h-6"
                      onError={(e) => {
                        // Hide broken image
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  )}
                  <Title>{displayName}</Title>
                </div>
              </Card>

              <Card>
                <Text>Mode</Text>
                <div className="mt-2">
                  <Title>{guardrailData.litellm_params?.mode || "-"}</Title>
                  <Badge color={guardrailData.litellm_params?.default_on ? "green" : "gray"}>
                    {guardrailData.litellm_params?.default_on ? "Default On" : "Default Off"}
                  </Badge>
                </div>
              </Card>

              <Card>
                <Text>Created At</Text>
                <div className="mt-2">
                  <Title>{formatDate(guardrailData.created_at)}</Title>
                  <Text>Last Updated: {formatDate(guardrailData.updated_at)}</Text>
                </div>
              </Card>
            </Grid>

            <Card className="mt-6">
              <Text>Provider Configuration</Text>
              <div className="mt-2 space-y-2">
                {Object.entries(guardrailData.litellm_params || {}).map(([key, value]) => {
                  // Skip mode and guardrail as they're displayed above
                  if (key === 'mode' || key === 'guardrail' || key === 'default_on') return null;
                  
                  return (
                    <div key={key} className="flex">
                      <Text className="font-medium w-1/3">{key}</Text>
                      <Text className="w-2/3">
                        {typeof value === 'object' 
                          ? JSON.stringify(value, null, 2) 
                          : String(value)}
                      </Text>
                    </div>
                  );
                })}
              </div>
            </Card>

            {guardrailData.guardrail_info && Object.keys(guardrailData.guardrail_info).length > 0 && (
              <Card className="mt-6">
                <Text>Additional Information</Text>
                <div className="mt-2 space-y-2">
                  {Object.entries(guardrailData.guardrail_info).map(([key, value]) => (
                    <div key={key} className="flex">
                      <Text className="font-medium w-1/3">{key}</Text>
                      <Text className="w-2/3">
                        {typeof value === 'object' 
                          ? JSON.stringify(value, null, 2) 
                          : String(value)}
                      </Text>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </TabPanel>

          {/* Settings Panel (only for admins) */}
          {isAdmin && (
            <TabPanel>
              <Card>
                <div className="flex justify-between items-center mb-4">
                  <Title>Guardrail Settings</Title>
                  {!isEditing && (
                    <TremorButton 
                      onClick={() => setIsEditing(true)}
                    >
                      Edit Settings
                    </TremorButton>
                  )}
                </div>

                {isEditing ? (
                  <Form
                    form={form}
                    onFinish={handleGuardrailUpdate}
                    initialValues={{
                      guardrail_name: guardrailData.guardrail_name,
                      ...guardrailData.litellm_params,
                      guardrail_info: guardrailData.guardrail_info 
                        ? JSON.stringify(guardrailData.guardrail_info, null, 2) 
                        : "",
                    }}
                    layout="vertical"
                  >
                    <Form.Item
                      label="Guardrail Name"
                      name="guardrail_name"
                      rules={[{ required: true, message: "Please input a guardrail name" }]}
                    >
                      <Input />
                    </Form.Item>
                    
                    <Form.Item
                      label="Default On"
                      name="default_on"
                    >
                      <Select>
                        <Select.Option value={true}>Yes</Select.Option>
                        <Select.Option value={false}>No</Select.Option>
                      </Select>
                    </Form.Item>

                    <Form.Item
                      label="Additional Information"
                      name="guardrail_info"
                    >
                      <Input.TextArea rows={5} />
                    </Form.Item>

                    <div className="flex justify-end gap-2 mt-6">
                      <Button onClick={() => setIsEditing(false)}>
                        Cancel
                      </Button>
                      <TremorButton>
                        Save Changes
                      </TremorButton>
                    </div>
                  </Form>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <Text className="font-medium">Guardrail ID</Text>
                      <div className="font-mono">{guardrailData.guardrail_id}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Guardrail Name</Text>
                      <div>{guardrailData.guardrail_name || "Unnamed Guardrail"}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Provider</Text>
                      <div>{displayName}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Mode</Text>
                      <div>{guardrailData.litellm_params?.mode || "-"}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Default On</Text>
                      <Badge color={guardrailData.litellm_params?.default_on ? "green" : "gray"}>
                        {guardrailData.litellm_params?.default_on ? "Yes" : "No"}
                      </Badge>
                    </div>
                    <div>
                      <Text className="font-medium">Created At</Text>
                      <div>{formatDate(guardrailData.created_at)}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Last Updated</Text>
                      <div>{formatDate(guardrailData.updated_at)}</div>
                    </div>
                  </div>
                )}
              </Card>
            </TabPanel>
          )}
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default GuardrailInfoView; 