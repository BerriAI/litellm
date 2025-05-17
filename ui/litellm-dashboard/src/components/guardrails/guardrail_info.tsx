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
  TextInput,
} from "@tremor/react";
import { Button, Form, Input, Select, message, Tooltip, Divider } from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { getGuardrailInfo, updateGuardrailCall, getGuardrailUISettings } from "@/components/networking";
import { getGuardrailLogoAndName } from "./guardrail_info_helpers";
import PiiConfiguration from "./pii_configuration";

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
  const [selectedPiiEntities, setSelectedPiiEntities] = useState<string[]>([]);
  const [selectedPiiActions, setSelectedPiiActions] = useState<{[key: string]: string}>({});
  const [guardrailSettings, setGuardrailSettings] = useState<{
    supported_entities: string[];
    supported_actions: string[];
    pii_entity_categories: Array<{
      category: string;
      entities: string[];
    }>;
    supported_modes: string[];
  } | null>(null);

  const fetchGuardrailInfo = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await getGuardrailInfo(accessToken, guardrailId);
      setGuardrailData(response);
      
      // Initialize PII configuration from guardrail data
      if (response.litellm_params?.pii_entities_config) {
        const piiConfig = response.litellm_params.pii_entities_config;
        
        // Clear previous selections
        setSelectedPiiEntities([]);
        setSelectedPiiActions({});
        
        // Only if there are entities configured
        if (Object.keys(piiConfig).length > 0) {
          const entities: string[] = [];
          const actions: {[key: string]: string} = {};
          
          Object.entries(piiConfig).forEach(([entity, action]: [string, any]) => {
            entities.push(entity);
            actions[entity] = typeof action === 'string' ? action : "MASK";
          });
          
          setSelectedPiiEntities(entities);
          setSelectedPiiActions(actions);
        }
      } else {
        // Clear selections if no PII config exists
        setSelectedPiiEntities([]);
        setSelectedPiiActions({});
      }
    } catch (error) {
      message.error("Failed to load guardrail information");
      console.error("Error fetching guardrail info:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchGuardrailUISettings = async () => {
    try {
      if (!accessToken) return;
      const uiSettings = await getGuardrailUISettings(accessToken);
      setGuardrailSettings(uiSettings);
    } catch (error) {
      console.error("Error fetching guardrail UI settings:", error);
    }
  };

  useEffect(() => {
    fetchGuardrailInfo();
    fetchGuardrailUISettings();
  }, [guardrailId, accessToken]);

  const handlePiiEntitySelect = (entity: string) => {
    setSelectedPiiEntities(prev => {
      if (prev.includes(entity)) {
        return prev.filter(e => e !== entity);
      } else {
        return [...prev, entity];
      }
    });
  };

  const handlePiiActionSelect = (entity: string, action: string) => {
    setSelectedPiiActions(prev => ({
      ...prev,
      [entity]: action
    }));
  };

  const handleGuardrailUpdate = async (values: any) => {
    try {
      if (!accessToken) return;
      
      // Prepare update data object
      const updateData: any = {
        guardrail_name: values.guardrail_name,
        litellm_params: {
          default_on: values.default_on,
        },
        guardrail_info: values.guardrail_info ? JSON.parse(values.guardrail_info) : undefined
      };
      
      // Only add PII entities config if we have selected entities
      if (selectedPiiEntities.length > 0) {
        // Create PII config object only with selected entities
        const piiEntitiesConfig: {[key: string]: string} = {};
        selectedPiiEntities.forEach(entity => {
          piiEntitiesConfig[entity] = selectedPiiActions[entity] || "MASK";
        });
        
        // Add to litellm_params only if we have entities
        updateData.litellm_params.pii_entities_config = piiEntitiesConfig;
      } else {
        // If no entities selected, explicitly set to empty object
        // This will clear any existing PII config
        updateData.litellm_params.pii_entities_config = {};
      }
      
      await updateGuardrailCall(accessToken, guardrailId, updateData);
      message.success("Guardrail updated successfully");
      fetchGuardrailInfo();
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating guardrail:", error);
      message.error("Failed to update guardrail");
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

            {guardrailData.litellm_params?.pii_entities_config && Object.keys(guardrailData.litellm_params.pii_entities_config).length > 0 && (
              <Card className="mt-6">
                <div className="flex justify-between items-center">
                  <Text className="font-medium">PII Protection</Text>
                  <Badge color="blue">
                    {Object.keys(guardrailData.litellm_params.pii_entities_config).length} PII entities configured
                  </Badge>
                </div>
              </Card>
            )}

            {guardrailData.guardrail_info && Object.keys(guardrailData.guardrail_info).length > 0 && (
              <Card className="mt-6">
                <Text>Guardrail Info</Text>
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
                      <TextInput />
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
                    
                    <Divider orientation="left">PII Protection</Divider>
                    <div className="mb-6">
                      {guardrailSettings && (
                        <PiiConfiguration 
                          entities={guardrailSettings.supported_entities}
                          actions={guardrailSettings.supported_actions}
                          selectedEntities={selectedPiiEntities}
                          selectedActions={selectedPiiActions}
                          onEntitySelect={handlePiiEntitySelect}
                          onActionSelect={handlePiiActionSelect}
                          entityCategories={guardrailSettings.pii_entity_categories}
                        />
                      )}
                    </div>

                    <Divider orientation="left">Advanced Settings</Divider>
                    <Form.Item
                      label="Guardrail Information"
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
                    
                    {guardrailData.litellm_params?.pii_entities_config && Object.keys(guardrailData.litellm_params.pii_entities_config).length > 0 && (
                      <div>
                        <Text className="font-medium">PII Protection</Text>
                        <div className="mt-2">
                          <Badge color="blue">
                            {Object.keys(guardrailData.litellm_params.pii_entities_config).length} PII entities configured
                          </Badge>
                        </div>
                      </div>
                    )}
                    
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