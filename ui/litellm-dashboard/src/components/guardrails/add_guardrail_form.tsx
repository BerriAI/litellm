import React, { useState, useEffect } from "react";
import { Form, Typography, Select, Modal, Tag, Steps } from "antd";
import { Button, TextInput } from "@tremor/react";
import {
  guardrail_provider_map,
  shouldRenderPIIConfigSettings,
  guardrailLogoMap,
  populateGuardrailProviders,
  populateGuardrailProviderMap,
  getGuardrailProviders,
} from "./guardrail_info_helpers";
import { createGuardrailCall, getGuardrailUISettings, getGuardrailProviderSpecificParams } from "../networking";
import PiiConfiguration from "./pii_configuration";
import GuardrailProviderFields from "./guardrail_provider_fields";
import GuardrailOptionalParams from "./guardrail_optional_params";
import NotificationsManager from "../molecules/notifications_manager";

const { Title, Text, Link } = Typography;
const { Option } = Select;
const { Step } = Steps;

// Define human-friendly descriptions for each mode
const modeDescriptions = {
  pre_call: "Before LLM Call - Runs before the LLM call and checks the input (Recommended)",
  during_call: "During LLM Call - Runs in parallel with the LLM call, with response held until check completes",
  post_call: "After LLM Call - Runs after the LLM call and checks only the output",
  logging_only: "Logging Only - Only runs on logging callbacks without affecting the LLM call",
  pre_mcp_call: "Before MCP Tool Call - Runs before MCP tool execution and validates tool calls",
  during_mcp_call: "During MCP Tool Call - Runs in parallel with MCP tool execution for monitoring",
};

interface AddGuardrailFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

interface GuardrailSettings {
  supported_entities: string[];
  supported_actions: string[];
  supported_modes: string[];
  pii_entity_categories: Array<{
    category: string;
    entities: string[];
  }>;
}

interface LiteLLMParams {
  guardrail: string;
  mode: string;
  default_on: boolean;
  [key: string]: any; // Allow additional properties for specific guardrails
}

// Mapping of provider -> list of param descriptors
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

const AddGuardrailForm: React.FC<AddGuardrailFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [guardrailSettings, setGuardrailSettings] = useState<GuardrailSettings | null>(null);
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [selectedActions, setSelectedActions] = useState<{ [key: string]: string }>({});
  const [currentStep, setCurrentStep] = useState(0);
  const [providerParams, setProviderParams] = useState<ProviderParamsResponse | null>(null);

  // Azure Text Moderation state
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [globalSeverityThreshold, setGlobalSeverityThreshold] = useState<number>(2);
  const [categorySpecificThresholds, setCategorySpecificThresholds] = useState<{ [key: string]: number }>({});

  // Fetch guardrail UI settings + provider params on mount / accessToken change
  useEffect(() => {
    if (!accessToken) return;

    const fetchData = async () => {
      try {
        // Parallel requests for speed
        const [uiSettings, providerParamsResp] = await Promise.all([
          getGuardrailUISettings(accessToken),
          getGuardrailProviderSpecificParams(accessToken),
        ]);

        setGuardrailSettings(uiSettings);
        setProviderParams(providerParamsResp);

        // Populate dynamic providers from API response
        populateGuardrailProviders(providerParamsResp);
        populateGuardrailProviderMap(providerParamsResp);
      } catch (error) {
        console.error("Error fetching guardrail data:", error);
        NotificationsManager.fromBackend("Failed to load guardrail configuration");
      }
    };

    fetchData();
  }, [accessToken]);

  const handleProviderChange = (value: string) => {
    setSelectedProvider(value);
    // Reset form fields that are provider-specific
    form.setFieldsValue({
      config: undefined,
      presidio_analyzer_api_base: undefined,
      presidio_anonymizer_api_base: undefined,
    });

    // Reset PII selections when changing provider
    setSelectedEntities([]);
    setSelectedActions({});

    // Reset Azure Text Moderation selections when changing provider
    setSelectedCategories([]);
    setGlobalSeverityThreshold(2);
    setCategorySpecificThresholds({});
  };

  const handleEntitySelect = (entity: string) => {
    setSelectedEntities((prev) => {
      if (prev.includes(entity)) {
        return prev.filter((e) => e !== entity);
      } else {
        return [...prev, entity];
      }
    });
  };

  const handleActionSelect = (entity: string, action: string) => {
    setSelectedActions((prev) => ({
      ...prev,
      [entity]: action,
    }));
  };

  // Azure Text Moderation handlers
  const handleCategorySelect = (category: string) => {
    setSelectedCategories((prev) =>
      prev.includes(category) ? prev.filter((c) => c !== category) : [...prev, category],
    );
  };

  const handleGlobalSeverityChange = (threshold: number) => {
    setGlobalSeverityThreshold(threshold);
  };

  const handleCategorySeverityChange = (category: string, threshold: number) => {
    setCategorySpecificThresholds((prev) => ({
      ...prev,
      [category]: threshold,
    }));
  };

  const nextStep = async () => {
    try {
      // Validate current step fields
      if (currentStep === 0) {
        await form.validateFields(["guardrail_name", "provider", "mode", "default_on"]);
        // Also validate provider-specific fields if applicable
        if (selectedProvider) {
          // This will automatically validate any required fields for the selected provider
          const fieldsToValidate = ["guardrail_name", "provider", "mode", "default_on"];

          if (selectedProvider === "PresidioPII") {
            fieldsToValidate.push("presidio_analyzer_api_base", "presidio_anonymizer_api_base");
          }
          await form.validateFields(fieldsToValidate);
        }
      }

      // Validate configuration steps
      if (currentStep === 1) {
        if (shouldRenderPIIConfigSettings(selectedProvider) && selectedEntities.length === 0) {
          NotificationsManager.fromBackend("Please select at least one PII entity to continue");
          return;
        }
      }

      setCurrentStep(currentStep + 1);
    } catch (error) {
      console.error("Form validation failed:", error);
    }
  };

  const prevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  const resetForm = () => {
    form.resetFields();
    setSelectedProvider(null);
    setSelectedEntities([]);
    setSelectedActions({});
    setSelectedCategories([]);
    setGlobalSeverityThreshold(2);
    setCategorySpecificThresholds({});
    setCurrentStep(0);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      // First validate currently visible fields
      await form.validateFields();

      // After validation, fetch *all* form values (including those from previous steps)
      const values = form.getFieldsValue(true);

      // Get the guardrail provider value from the map
      const guardrailProvider = guardrail_provider_map[values.provider];

      // Prepare the guardrail data with proper typings
      const guardrailData: {
        guardrail_name: string;
        litellm_params: {
          guardrail: string;
          mode: string;
          default_on: boolean;
          [key: string]: any; // Allow dynamic properties
        };
        guardrail_info: any;
      } = {
        guardrail_name: values.guardrail_name,
        litellm_params: {
          guardrail: guardrailProvider,
          mode: values.mode,
          default_on: values.default_on,
        },
        guardrail_info: {},
      };

      // For Presidio PII, add the entity and action configurations
      if (values.provider === "PresidioPII" && selectedEntities.length > 0) {
        const piiEntitiesConfig: { [key: string]: string } = {};
        selectedEntities.forEach((entity) => {
          piiEntitiesConfig[entity] = selectedActions[entity] || "MASK"; // Default to MASK if no action selected
        });

        guardrailData.litellm_params.pii_entities_config = piiEntitiesConfig;

        // Add Presidio API bases if provided
        if (values.presidio_analyzer_api_base) {
          guardrailData.litellm_params.presidio_analyzer_api_base = values.presidio_analyzer_api_base;
        }
        if (values.presidio_anonymizer_api_base) {
          guardrailData.litellm_params.presidio_anonymizer_api_base = values.presidio_anonymizer_api_base;
        }
      }
      // Add config values to the guardrail_info if provided
      else if (values.config) {
        try {
          const configObj = JSON.parse(values.config);
          // For some guardrails, the config values need to be in litellm_params
          guardrailData.guardrail_info = configObj;
        } catch (error) {
          NotificationsManager.fromBackend("Invalid JSON in configuration");
          setLoading(false);
          return;
        }
      }

      /******************************
       * Add provider-specific params
       * ----------------------------------
       * The backend exposes exactly which extra parameters a provider
       * accepts via `/guardrails/ui/provider_specific_params`.
       * Instead of copying every unknown form field, we fetch the list for
       * the selected provider and ONLY pass those recognised params.
       ******************************/

      console.log("values: ", JSON.stringify(values));

      // Use pre-fetched provider params to copy recognised params
      if (providerParams && selectedProvider) {
        const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
        console.log("providerKey: ", providerKey);
        const providerSpecificParams = providerParams[providerKey] || {};

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

          if (paramValue !== undefined && paramValue !== null && paramValue !== "") {
            guardrailData.litellm_params[paramName] = paramValue;
          }
        });
      }

      if (!accessToken) {
        throw new Error("No access token available");
      }

      console.log("Sending guardrail data:", JSON.stringify(guardrailData));
      await createGuardrailCall(accessToken, guardrailData);

      NotificationsManager.success("Guardrail created successfully");

      // Reset form and close modal
      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create guardrail:", error);
      NotificationsManager.fromBackend(
        "Failed to create guardrail: " + (error instanceof Error ? error.message : String(error)),
      );
    } finally {
      setLoading(false);
    }
  };

  const renderBasicInfo = () => {
    return (
      <>
        <Form.Item
          name="guardrail_name"
          label="Guardrail Name"
          rules={[{ required: true, message: "Please enter a guardrail name" }]}
        >
          <TextInput placeholder="Enter a name for this guardrail" />
        </Form.Item>

        <Form.Item
          name="provider"
          label="Guardrail Provider"
          rules={[{ required: true, message: "Please select a provider" }]}
        >
          <Select
            placeholder="Select a guardrail provider"
            onChange={handleProviderChange}
            labelInValue={false}
            optionLabelProp="label"
            dropdownRender={(menu) => menu}
            showSearch={true}
          >
            {Object.entries(getGuardrailProviders()).map(([key, value]) => (
              <Option
                key={key}
                value={key}
                label={
                  <div style={{ display: "flex", alignItems: "center" }}>
                    {guardrailLogoMap[value] && (
                      <img
                        src={guardrailLogoMap[value]}
                        alt=""
                        style={{
                          height: "20px",
                          width: "20px",
                          marginRight: "8px",
                          objectFit: "contain",
                        }}
                        onError={(e) => {
                          // Hide broken image icon if image fails to load
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    )}
                    <span>{value}</span>
                  </div>
                }
              >
                <div style={{ display: "flex", alignItems: "center" }}>
                  {guardrailLogoMap[value] && (
                    <img
                      src={guardrailLogoMap[value]}
                      alt=""
                      style={{
                        height: "20px",
                        width: "20px",
                        marginRight: "8px",
                        objectFit: "contain",
                      }}
                      onError={(e) => {
                        // Hide broken image icon if image fails to load
                        e.currentTarget.style.display = "none";
                      }}
                    />
                  )}
                  <span>{value}</span>
                </div>
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          name="mode"
          label="Mode"
          tooltip="How the guardrail should be applied"
          rules={[{ required: true, message: "Please select a mode" }]}
        >
          <Select optionLabelProp="label" mode="multiple">
            {guardrailSettings?.supported_modes?.map((mode) => (
              <Option key={mode} value={mode} label={mode}>
                <div>
                  <div>
                    <strong>{mode}</strong>
                    {mode === "pre_call" && (
                      <Tag color="green" style={{ marginLeft: "8px" }}>
                        Recommended
                      </Tag>
                    )}
                  </div>
                  <div style={{ fontSize: "12px", color: "#888" }}>
                    {modeDescriptions[mode as keyof typeof modeDescriptions]}
                  </div>
                </div>
              </Option>
            )) || (
              <>
                <Option value="pre_call" label="pre_call">
                  <div>
                    <div>
                      <strong>pre_call</strong> <Tag color="green">Recommended</Tag>
                    </div>
                    <div style={{ fontSize: "12px", color: "#888" }}>{modeDescriptions.pre_call}</div>
                  </div>
                </Option>
                <Option value="during_call" label="during_call">
                  <div>
                    <div>
                      <strong>during_call</strong>
                    </div>
                    <div style={{ fontSize: "12px", color: "#888" }}>{modeDescriptions.during_call}</div>
                  </div>
                </Option>
                <Option value="post_call" label="post_call">
                  <div>
                    <div>
                      <strong>post_call</strong>
                    </div>
                    <div style={{ fontSize: "12px", color: "#888" }}>{modeDescriptions.post_call}</div>
                  </div>
                </Option>
                <Option value="logging_only" label="logging_only">
                  <div>
                    <div>
                      <strong>logging_only</strong>
                    </div>
                    <div style={{ fontSize: "12px", color: "#888" }}>{modeDescriptions.logging_only}</div>
                  </div>
                </Option>
              </>
            )}
          </Select>
        </Form.Item>

        <Form.Item
          name="default_on"
          label="Always On"
          tooltip="If enabled, this guardrail will be applied to all requests by default."
        >
          <Select>
            <Select.Option value={true}>Yes</Select.Option>
            <Select.Option value={false}>No</Select.Option>
          </Select>
        </Form.Item>

        {/* Use the GuardrailProviderFields component to render provider-specific fields */}
        <GuardrailProviderFields
          selectedProvider={selectedProvider}
          accessToken={accessToken}
          providerParams={providerParams}
        />
      </>
    );
  };

  const renderPiiConfiguration = () => {
    if (!guardrailSettings || selectedProvider !== "PresidioPII") return null;

    return (
      <PiiConfiguration
        entities={guardrailSettings.supported_entities}
        actions={guardrailSettings.supported_actions}
        selectedEntities={selectedEntities}
        selectedActions={selectedActions}
        onEntitySelect={handleEntitySelect}
        onActionSelect={handleActionSelect}
        entityCategories={guardrailSettings.pii_entity_categories}
      />
    );
  };

  const renderOptionalParams = () => {
    if (!selectedProvider || !providerParams) return null;

    console.log("guardrail_provider_map: ", guardrail_provider_map);
    console.log("selectedProvider: ", selectedProvider);
    const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
    const providerFields = providerParams && providerParams[providerKey];

    if (!providerFields || !providerFields.optional_params) return null;

    return <GuardrailOptionalParams optionalParams={providerFields.optional_params} parentFieldKey="optional_params" />;
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return renderBasicInfo();
      case 1:
        if (shouldRenderPIIConfigSettings(selectedProvider)) {
          return renderPiiConfiguration();
        }
        return renderOptionalParams();
      default:
        return null;
    }
  };

  const renderStepButtons = () => {
    return (
      <div className="flex justify-end space-x-2 mt-4">
        {currentStep > 0 && (
          <Button variant="secondary" onClick={prevStep}>
            Previous
          </Button>
        )}
        {currentStep < 2 && <Button onClick={nextStep}>Next</Button>}
        {currentStep === 2 && (
          <Button onClick={handleSubmit} loading={loading}>
            Create Guardrail
          </Button>
        )}
        <Button variant="secondary" onClick={handleClose}>
          Cancel
        </Button>
      </div>
    );
  };

  return (
    <Modal title="Add Guardrail" open={visible} onCancel={handleClose} footer={null} width={700}>
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          mode: "pre_call",
          default_on: false,
        }}
      >
        <Steps current={currentStep} className="mb-6">
          <Step title="Basic Info" />
          <Step
            title={shouldRenderPIIConfigSettings(selectedProvider) ? "PII Configuration" : "Provider Configuration"}
          />
        </Steps>

        {renderStepContent()}
        {renderStepButtons()}
      </Form>
    </Modal>
  );
};

export default AddGuardrailForm;
