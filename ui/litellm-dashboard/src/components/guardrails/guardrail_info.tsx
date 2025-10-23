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
import { Button, Form, Input, Select, Divider } from "antd";
import {
  getGuardrailInfo,
  updateGuardrailCall,
  getGuardrailUISettings,
  getGuardrailProviderSpecificParams,
} from "@/components/networking";
import { getGuardrailLogoAndName, guardrail_provider_map } from "./guardrail_info_helpers";
import PiiConfiguration from "./pii_configuration";
import GuardrailProviderFields from "./guardrail_provider_fields";
import GuardrailOptionalParams from "./guardrail_optional_params";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CheckIcon, CopyIcon } from "lucide-react";
import NotificationsManager from "../molecules/notifications_manager";

export interface GuardrailInfoProps {
  guardrailId: string;
  onClose: () => void;
  accessToken: string | null;
  isAdmin: boolean;
}

interface ProviderParam {
  param: string;
  description: string;
  required: boolean;
  default_value?: string;
  options?: string[];
  type?: string;
  fields?: { [key: string]: ProviderParam };
  dict_key_options?: string[];
  dict_value_type?: string;
}

interface ProviderParamsResponse {
  [provider: string]: { [key: string]: ProviderParam };
}

const GuardrailInfoView: React.FC<GuardrailInfoProps> = ({ guardrailId, onClose, accessToken, isAdmin }) => {
  const [guardrailData, setGuardrailData] = useState<any>(null);
  const [guardrailProviderSpecificParams, setGuardrailProviderSpecificParams] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();
  const [selectedPiiEntities, setSelectedPiiEntities] = useState<string[]>([]);
  const [selectedPiiActions, setSelectedPiiActions] = useState<{ [key: string]: string }>({});
  const [guardrailSettings, setGuardrailSettings] = useState<{
    supported_entities: string[];
    supported_actions: string[];
    pii_entity_categories: Array<{
      category: string;
      entities: string[];
    }>;
    supported_modes: string[];
  } | null>(null);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
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
          const actions: { [key: string]: string } = {};

          Object.entries(piiConfig).forEach(([entity, action]: [string, any]) => {
            entities.push(entity);
            actions[entity] = typeof action === "string" ? action : "MASK";
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
      NotificationsManager.fromBackend("Failed to load guardrail information");
      console.error("Error fetching guardrail info:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchGuardrailProviderSpecificParams = async () => {
    try {
      if (!accessToken) return;
      const response = await getGuardrailProviderSpecificParams(accessToken);
      setGuardrailProviderSpecificParams(response);
    } catch (error) {
      console.error("Error fetching guardrail provider specific params:", error);
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
    fetchGuardrailProviderSpecificParams();
  }, [accessToken]);

  useEffect(() => {
    fetchGuardrailInfo();
    fetchGuardrailUISettings();
  }, [guardrailId, accessToken]);

  // Reset form when guardrail data or provider params change
  useEffect(() => {
    if (guardrailData && form) {
      form.setFieldsValue({
        guardrail_name: guardrailData.guardrail_name,
        ...guardrailData.litellm_params,
        guardrail_info: guardrailData.guardrail_info ? JSON.stringify(guardrailData.guardrail_info, null, 2) : "",
        // Include any optional_params if they exist
        ...(guardrailData.litellm_params?.optional_params && {
          optional_params: guardrailData.litellm_params.optional_params,
        }),
      });
    }
  }, [guardrailData, guardrailProviderSpecificParams, form]);

  const handlePiiEntitySelect = (entity: string) => {
    setSelectedPiiEntities((prev) => {
      if (prev.includes(entity)) {
        return prev.filter((e) => e !== entity);
      } else {
        return [...prev, entity];
      }
    });
  };

  const handlePiiActionSelect = (entity: string, action: string) => {
    setSelectedPiiActions((prev) => ({
      ...prev,
      [entity]: action,
    }));
  };

  const handleGuardrailUpdate = async (values: any) => {
    try {
      if (!accessToken) return;

      // Prepare update data object - only include changed fields
      const updateData: any = {
        litellm_params: {},
      };

      // Only include guardrail_name if it has changed
      if (values.guardrail_name !== guardrailData.guardrail_name) {
        updateData.guardrail_name = values.guardrail_name;
      }

      // Only include default_on if it has changed
      if (values.default_on !== guardrailData.litellm_params?.default_on) {
        updateData.litellm_params.default_on = values.default_on;
      }

      // Only include guardrail_info if it has changed
      const originalGuardrailInfo = guardrailData.guardrail_info;
      const newGuardrailInfo = values.guardrail_info ? JSON.parse(values.guardrail_info) : undefined;
      if (JSON.stringify(originalGuardrailInfo) !== JSON.stringify(newGuardrailInfo)) {
        updateData.guardrail_info = newGuardrailInfo;
      }

      // Only add PII entities config if there are changes
      const originalPiiConfig = guardrailData.litellm_params?.pii_entities_config || {};
      const newPiiEntitiesConfig: { [key: string]: string } = {};

      selectedPiiEntities.forEach((entity) => {
        newPiiEntitiesConfig[entity] = selectedPiiActions[entity] || "MASK";
      });

      // Only update if PII config has changed
      if (JSON.stringify(originalPiiConfig) !== JSON.stringify(newPiiEntitiesConfig)) {
        updateData.litellm_params.pii_entities_config = newPiiEntitiesConfig;
      }

      /******************************
       * Add provider-specific params (reusing logic from add_guardrail_form.tsx)
       * ----------------------------------
       * The backend exposes exactly which extra parameters a provider
       * accepts via `/guardrails/ui/provider_specific_params`.
       * Instead of copying every unknown form field, we fetch the list for
       * the selected provider and ONLY pass those recognised params.
       ******************************/

      // Get the current provider from the guardrail data
      const currentProvider = Object.keys(guardrail_provider_map).find(
        (key) => guardrail_provider_map[key] === guardrailData.litellm_params?.guardrail,
      );

      console.log("values: ", JSON.stringify(values));
      console.log("currentProvider: ", currentProvider);

      // Use pre-fetched provider params to copy recognised params
      if (guardrailProviderSpecificParams && currentProvider) {
        const providerKey = guardrail_provider_map[currentProvider]?.toLowerCase();
        const providerSpecificParams = guardrailProviderSpecificParams[providerKey] || {};

        const allowedParams = new Set<string>();

        console.log("providerSpecificParams: ", JSON.stringify(providerSpecificParams));

        // Add root-level parameters (like api_key, api_base, api_version)
        Object.keys(providerSpecificParams).forEach((paramName) => {
          if (paramName !== "optional_params") {
            allowedParams.add(paramName);
          }
        });

        // Add nested parameters from optional_params.fields
        if (providerSpecificParams.optional_params && providerSpecificParams.optional_params.fields) {
          Object.keys(providerSpecificParams.optional_params.fields).forEach((paramName) => {
            allowedParams.add(paramName);
          });
        }

        console.log("allowedParams: ", allowedParams);
        allowedParams.forEach((paramName) => {
          // Check for both direct parameter name and nested optional_params object
          let paramValue = values[paramName];
          if (paramValue === undefined || paramValue === null || paramValue === "") {
            paramValue = values.optional_params?.[paramName];
          }

          // Get the original value for comparison
          const originalValue = guardrailData.litellm_params?.[paramName];

          // Check if the value has changed from the original
          const hasChanged = JSON.stringify(paramValue) !== JSON.stringify(originalValue);

          // Include if value has changed and has a meaningful value, OR if user explicitly cleared a value
          if (hasChanged) {
            if (paramValue !== undefined && paramValue !== null && paramValue !== "") {
              // User set a new value
              updateData.litellm_params[paramName] = paramValue;
            } else if (originalValue !== undefined && originalValue !== null && originalValue !== "") {
              // User cleared an existing value - set to null to indicate removal
              updateData.litellm_params[paramName] = null;
            }
          }
        });
      }

      // Remove empty litellm_params object if no parameters were changed
      if (Object.keys(updateData.litellm_params).length === 0) {
        delete updateData.litellm_params;
      }

      // Only proceed with update if there are actual changes
      if (Object.keys(updateData).length === 0) {
        NotificationsManager.info("No changes detected");
        setIsEditing(false);
        return;
      }

      await updateGuardrailCall(accessToken, guardrailId, updateData);
      NotificationsManager.success("Guardrail updated successfully");
      fetchGuardrailInfo();
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating guardrail:", error);
      NotificationsManager.fromBackend("Failed to update guardrail");
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

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  return (
    <div className="p-4">
      <div>
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Guardrails
        </TremorButton>
        <Title>{guardrailData.guardrail_name || "Unnamed Guardrail"}</Title>
        <div className="flex items-center cursor-pointer">
          <Text className="text-gray-500 font-mono">{guardrailData.guardrail_id}</Text>

          <Button
            type="text"
            size="small"
            icon={copiedStates["guardrail-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
            onClick={() => copyToClipboard(guardrailData.guardrail_id, "guardrail-id")}
            className={`left-2 z-10 transition-all duration-200 ${
              copiedStates["guardrail-id"]
                ? "text-green-600 bg-green-50 border-green-200"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
            }`}
          />
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
                        (e.target as HTMLImageElement).style.display = "none";
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

            {guardrailData.litellm_params?.pii_entities_config &&
              Object.keys(guardrailData.litellm_params.pii_entities_config).length > 0 && (
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
                        {typeof value === "object" ? JSON.stringify(value, null, 2) : String(value)}
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
                  {!isEditing && <TremorButton onClick={() => setIsEditing(true)}>Edit Settings</TremorButton>}
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
                      // Include any optional_params if they exist
                      ...(guardrailData.litellm_params?.optional_params && {
                        optional_params: guardrailData.litellm_params.optional_params,
                      }),
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

                    <Form.Item label="Default On" name="default_on">
                      <Select>
                        <Select.Option value={true}>Yes</Select.Option>
                        <Select.Option value={false}>No</Select.Option>
                      </Select>
                    </Form.Item>

                    {guardrailData.litellm_params?.guardrail === "presidio" && (
                      <>
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
                      </>
                    )}

                    <Divider orientation="left">Provider Settings</Divider>

                    {/* Provider-specific fields */}
                    <GuardrailProviderFields
                      selectedProvider={
                        Object.keys(guardrail_provider_map).find(
                          (key) => guardrail_provider_map[key] === guardrailData.litellm_params?.guardrail,
                        ) || null
                      }
                      accessToken={accessToken}
                      providerParams={guardrailProviderSpecificParams}
                      value={guardrailData.litellm_params}
                    />

                    {/* Optional parameters */}
                    {guardrailProviderSpecificParams &&
                      (() => {
                        const currentProvider = Object.keys(guardrail_provider_map).find(
                          (key) => guardrail_provider_map[key] === guardrailData.litellm_params?.guardrail,
                        );
                        if (!currentProvider) return null;

                        const providerKey = guardrail_provider_map[currentProvider]?.toLowerCase();
                        const providerFields = guardrailProviderSpecificParams[providerKey];

                        if (!providerFields || !providerFields.optional_params) return null;

                        return (
                          <GuardrailOptionalParams
                            optionalParams={providerFields.optional_params}
                            parentFieldKey="optional_params"
                            values={guardrailData.litellm_params}
                          />
                        );
                      })()}

                    <Divider orientation="left">Advanced Settings</Divider>
                    <Form.Item label="Guardrail Information" name="guardrail_info">
                      <Input.TextArea rows={5} />
                    </Form.Item>

                    <div className="flex justify-end gap-2 mt-6">
                      <Button onClick={() => setIsEditing(false)}>Cancel</Button>
                      <TremorButton>Save Changes</TremorButton>
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

                    {guardrailData.litellm_params?.pii_entities_config &&
                      Object.keys(guardrailData.litellm_params.pii_entities_config).length > 0 && (
                        <div>
                          <Text className="font-medium">PII Protection</Text>
                          <div className="mt-2">
                            <Badge color="blue">
                              {Object.keys(guardrailData.litellm_params.pii_entities_config).length} PII entities
                              configured
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
