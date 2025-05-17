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
import { Button, Form, Input, Select, message, Tooltip } from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { getGuardrailInfo, updateGuardrailCall } from "@/components/networking";
import { getGuardrailLogoAndName } from "./guardrail_info_helpers";
import PiiConfiguration from "./pii_configuration";

// Available PII actions
const PII_ACTIONS = ["MASK", "BLOCK"];

// PII entity categories for organization
const PII_ENTITY_CATEGORIES = [
  {
    category: "Personal",
    entities: ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON_NAME"]
  },
  {
    category: "Financial",
    entities: ["CREDIT_CARD", "BANK_ACCOUNT", "NRP"]
  },
  {
    category: "Location",
    entities: ["LOCATION", "ADDRESS", "IP_ADDRESS", "URL"]
  },
  {
    category: "Government",
    entities: ["AU_ABN", "US_SSN", "UK_NHS"]
  }
];

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
  
  // For PII configuration
  const [piiEntities, setPiiEntities] = useState<string[]>([]);
  const [selectedPiiEntities, setSelectedPiiEntities] = useState<string[]>([]);
  const [selectedPiiActions, setSelectedPiiActions] = useState<{[key: string]: string}>({});

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

  // Set up PII entities and selected actions when guardrail data changes
  useEffect(() => {
    if (guardrailData?.guardrail_info?.pii_entities_config) {
      const piiConfig = guardrailData.guardrail_info.pii_entities_config;
      
      // Get unique entities from both categories and existing config
      const allEntities = new Set<string>();
      
      // Add entities from categories
      PII_ENTITY_CATEGORIES.forEach(category => {
        category.entities.forEach(entity => allEntities.add(entity));
      });
      
      // Add entities from existing config
      Object.keys(piiConfig).forEach(entity => allEntities.add(entity));
      
      setPiiEntities(Array.from(allEntities));
      setSelectedPiiEntities(Object.keys(piiConfig));
      setSelectedPiiActions(piiConfig);
    } else {
      // Default empty state
      const allEntities = new Set<string>();
      PII_ENTITY_CATEGORIES.forEach(category => {
        category.entities.forEach(entity => allEntities.add(entity));
      });
      
      setPiiEntities(Array.from(allEntities));
      setSelectedPiiEntities([]);
      setSelectedPiiActions({});
    }
  }, [guardrailData]);

  const handlePiiEntitySelect = (entity: string) => {
    setSelectedPiiEntities(prev => {
      if (prev.includes(entity)) {
        // Remove entity from selected list
        const newSelected = prev.filter(e => e !== entity);
        // Also remove from actions
        const newActions = {...selectedPiiActions};
        delete newActions[entity];
        setSelectedPiiActions(newActions);
        return newSelected;
      } else {
        // Add entity to selected list with default action of MASK
        const newActions = {...selectedPiiActions, [entity]: "MASK"};
        setSelectedPiiActions(newActions);
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
      
      // Create guardrail_info with PII configuration if entities selected
      const guardrailInfo = values.guardrail_info ? JSON.parse(values.guardrail_info) : {};
      
      // Add PII entities configuration
      if (selectedPiiEntities.length > 0) {
        guardrailInfo.pii_entities_config = {};
        selectedPiiEntities.forEach(entity => {
          guardrailInfo.pii_entities_config[entity] = selectedPiiActions[entity] || "MASK";
        });
      }
      
      const updateData = {
        guardrail_name: values.guardrail_name,
        litellm_params: {
          default_on: values.default_on
        },
        guardrail_info: guardrailInfo
      };
      
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

            {/* PII Configuration Display */}
            {guardrailData.guardrail_info?.pii_entities_config && 
             Object.keys(guardrailData.guardrail_info.pii_entities_config).length > 0 && (
              <Card className="mt-6">
                <Text>PII Entity Protection</Text>
                <div className="mt-2 space-y-2">
                  <div className="grid grid-cols-2 gap-4">
                    {Object.entries(guardrailData.guardrail_info.pii_entities_config).map(([entity, action]) => (
                      <div key={entity} className="flex items-center p-2 border rounded">
                        <Text className="font-medium flex-1">{entity.replace(/_/g, ' ')}</Text>
                                                 <Badge 
                           color={action === "MASK" ? "blue" : "red"}
                           className="px-2 py-1 rounded"
                         >
                           {String(action)}
                         </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            )}

            {guardrailData.guardrail_info && Object.keys(guardrailData.guardrail_info).length > 0 && (
              <Card className="mt-6">
                <Text>Guardrail Info</Text>
                <div className="mt-2 space-y-2">
                  {Object.entries(guardrailData.guardrail_info).map(([key, value]) => {
                    if (key === "pii_entities_config") return null; // Skip, already shown above
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
                  <div>
                    <Form
                      form={form}
                      onFinish={handleGuardrailUpdate}
                      initialValues={{
                        guardrail_name: guardrailData.guardrail_name,
                        ...guardrailData.litellm_params,
                        guardrail_info: guardrailData.guardrail_info 
                          ? JSON.stringify(Object.fromEntries(
                              Object.entries(guardrailData.guardrail_info)
                                .filter(([key]) => key !== "pii_entities_config")
                            ), null, 2)
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
                      
                      {/* PII Configuration */}
                      <Form.Item
                        label="PII Entity Protection"
                        help="Select PII entities to protect and choose an action for each entity"
                      >
                        <PiiConfiguration
                          entities={piiEntities}
                          actions={PII_ACTIONS}
                          selectedEntities={selectedPiiEntities}
                          selectedActions={selectedPiiActions}
                          onEntitySelect={handlePiiEntitySelect}
                          onActionSelect={handlePiiActionSelect}
                          entityCategories={PII_ENTITY_CATEGORIES}
                        />
                      </Form.Item>

                      <Form.Item
                        label="Additional Guardrail Information"
                        name="guardrail_info"
                        help="Enter any additional guardrail configuration in JSON format"
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
                  </div>
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