import React, { useState, useEffect } from "react";
import { Form, Typography, Select, Input, Switch, Modal } from "antd";
import { Button, TextInput } from "@tremor/react";
import {
  guardrail_provider_map,
  guardrailLogoMap,
  getGuardrailProviders,
  type SkipSystemMessageChoice,
} from "./guardrail_info_helpers";
import { getGuardrailUISettings, getGlobalLitellmHeaderName } from "../networking";
import PiiConfiguration from "./pii_configuration";
import NotificationsManager from "../molecules/notifications_manager";

const { Title, Text } = Typography;
const { Option } = Select;

interface EditGuardrailFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
  guardrailId: string;
  /** Full stored params merged into PUT so optional fields (e.g. content filter) are preserved. */
  fullLitellmParams?: Record<string, any> | null;
  initialValues: {
    guardrail_name: string;
    provider: string;
    mode: string;
    default_on: boolean;
    pii_entities_config?: { [key: string]: string };
    skip_system_message_choice?: SkipSystemMessageChoice;
    [key: string]: any;
  };
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

const EditGuardrailForm: React.FC<EditGuardrailFormProps> = ({
  visible,
  onClose,
  accessToken,
  onSuccess,
  guardrailId,
  fullLitellmParams,
  initialValues,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(initialValues?.provider || null);
  const [guardrailSettings, setGuardrailSettings] = useState<GuardrailSettings | null>(null);
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [selectedActions, setSelectedActions] = useState<{ [key: string]: string }>({});

  // Fetch guardrail settings when the component mounts
  useEffect(() => {
    const fetchGuardrailSettings = async () => {
      try {
        if (!accessToken) return;

        const data = await getGuardrailUISettings(accessToken);
        setGuardrailSettings(data);
      } catch (error) {
        console.error("Error fetching guardrail settings:", error);
        NotificationsManager.fromBackend("Failed to load guardrail settings");
      }
    };

    fetchGuardrailSettings();
  }, [accessToken]);

  // Initialize selected entities and actions from initialValues
  useEffect(() => {
    if (initialValues?.pii_entities_config && Object.keys(initialValues.pii_entities_config).length > 0) {
      const entities = Object.keys(initialValues.pii_entities_config);
      setSelectedEntities(entities);
      setSelectedActions(initialValues.pii_entities_config);
    }
  }, [initialValues]);

  const handleProviderChange = (value: string) => {
    setSelectedProvider(value);
    // Reset form fields that are provider-specific
    form.setFieldsValue({
      config: undefined,
    });

    // Reset PII selections when changing provider
    setSelectedEntities([]);
    setSelectedActions({});
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

  const handleSubmit = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();

      // Get the guardrail provider value from the map
      const guardrailProvider = guardrail_provider_map[values.provider];

      const litellm_params: Record<string, any> =
        fullLitellmParams && typeof fullLitellmParams === "object" ? { ...fullLitellmParams } : {};

      litellm_params.guardrail = guardrailProvider;
      litellm_params.mode = values.mode;
      litellm_params.default_on = values.default_on;

      const skipChoice = values.skip_system_message_choice as SkipSystemMessageChoice | undefined;
      if (skipChoice === "yes") {
        litellm_params.skip_system_message_in_guardrail = true;
      } else if (skipChoice === "no") {
        litellm_params.skip_system_message_in_guardrail = false;
      } else {
        delete litellm_params.skip_system_message_in_guardrail;
      }

      let guardrail_info: any = {};

      // For Presidio PII, add the entity and action configurations
      if (values.provider === "PresidioPII" && selectedEntities.length > 0) {
        const piiEntitiesConfig: { [key: string]: string } = {};
        selectedEntities.forEach((entity) => {
          piiEntitiesConfig[entity] = selectedActions[entity] || "MASK"; // Default to MASK if no action selected
        });

        litellm_params.pii_entities_config = piiEntitiesConfig;
      }
      // Add config values to the guardrail_info if provided
      else if (values.config) {
        try {
          const configObj = JSON.parse(values.config);
          // For some guardrails, the config values need to be in litellm_params
          // Especially for providers like Bedrock that need guardrailIdentifier and guardrailVersion
          if (values.provider === "Bedrock" && configObj) {
            if (configObj.guardrail_id) {
              litellm_params.guardrailIdentifier = configObj.guardrail_id;
            }
            if (configObj.guardrail_version) {
              litellm_params.guardrailVersion = configObj.guardrail_version;
            }
          } else {
            // For other providers, add the config to guardrail_info
            guardrail_info = configObj;
          }
        } catch (error) {
          NotificationsManager.fromBackend("Invalid JSON in configuration");
          setLoading(false);
          return;
        }
      }

      const guardrailData: {
        guardrail_id: string;
        guardrail: {
          guardrail_name: string;
          litellm_params: Record<string, any>;
          guardrail_info: any;
        };
      } = {
        guardrail_id: guardrailId,
        guardrail: {
          guardrail_name: values.guardrail_name,
          litellm_params,
          guardrail_info,
        },
      };

      if (!accessToken) {
        throw new Error("No access token available");
      }

      console.log("Sending guardrail update data:", JSON.stringify(guardrailData));

      // Call the update endpoint
      const url = `/guardrails/${guardrailId}`;
      const response = await fetch(url, {
        method: "PUT",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(guardrailData),
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(errorData || "Failed to update guardrail");
      }

      NotificationsManager.success("Guardrail updated successfully");

      // Reset and close
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to update guardrail:", error);
      NotificationsManager.fromBackend(
        "Failed to update guardrail: " + (error instanceof Error ? error.message : String(error)),
      );
    } finally {
      setLoading(false);
    }
  };

  const renderPiiConfiguration = () => {
    if (!guardrailSettings || !selectedProvider || selectedProvider !== "PresidioPII") return null;

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

  const renderProviderSpecificFields = () => {
    if (!selectedProvider) return null;

    // For Presidio, we use the new PII configuration UI
    if (selectedProvider === "PresidioPII") {
      return renderPiiConfiguration();
    }

    switch (selectedProvider) {
      case "Aporia":
        return (
          <Form.Item label="Aporia Configuration" name="config" tooltip="JSON configuration for Aporia">
            <Input.TextArea
              rows={4}
              placeholder={`{
  "api_key": "your_aporia_api_key",
  "project_name": "your_project_name"
}`}
            />
          </Form.Item>
        );
      case "AimSecurity":
        return (
          <Form.Item label="Aim Security Configuration" name="config" tooltip="JSON configuration for Aim Security">
            <Input.TextArea
              rows={4}
              placeholder={`{
  "api_key": "your_aim_api_key"
}`}
            />
          </Form.Item>
        );
      case "Bedrock":
        return (
          <Form.Item
            label="Amazon Bedrock Configuration"
            name="config"
            tooltip="JSON configuration for Amazon Bedrock guardrails"
          >
            <Input.TextArea
              rows={4}
              placeholder={`{
  "guardrail_id": "your_guardrail_id",
  "guardrail_version": "your_guardrail_version"
}`}
            />
          </Form.Item>
        );
      case "GuardrailsAI":
        return (
          <Form.Item label="Guardrails.ai Configuration" name="config" tooltip="JSON configuration for Guardrails.ai">
            <Input.TextArea
              rows={4}
              placeholder={`{
  "api_key": "your_guardrails_api_key",
  "guardrail_id": "your_guardrail_id"
}`}
            />
          </Form.Item>
        );
      case "LakeraAI":
        return (
          <Form.Item label="Lakera AI Configuration" name="config" tooltip="JSON configuration for Lakera AI">
            <Input.TextArea
              rows={4}
              placeholder={`{
  "api_key": "your_lakera_api_key"
}`}
            />
          </Form.Item>
        );
      case "PromptInjection":
        return (
          <Form.Item
            label="Prompt Injection Configuration"
            name="config"
            tooltip="JSON configuration for prompt injection detection"
          >
            <Input.TextArea
              rows={4}
              placeholder={`{
  "threshold": 0.8
}`}
            />
          </Form.Item>
        );
      default:
        return (
          <Form.Item label="Custom Configuration" name="config" tooltip="JSON configuration for your custom guardrail">
            <Input.TextArea
              rows={4}
              placeholder={`{
  "key1": "value1",
  "key2": "value2"
}`}
            />
          </Form.Item>
        );
    }
  };

  return (
    <Modal title="Edit Guardrail" open={visible} onCancel={onClose} footer={null} width={700}>
      <Form form={form} layout="vertical" initialValues={initialValues}>
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
            disabled={true} // Disable changing provider in edit mode
            optionLabelProp="label"
          >
            {Object.entries(getGuardrailProviders()).map(([key, value]) => (
              <Option key={key} value={key} label={value}>
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
          <Select>
            {guardrailSettings?.supported_modes?.map((mode) => (
              <Option key={mode} value={mode}>
                {mode}
              </Option>
            )) || (
              <>
                <Option value="pre_call">pre_call</Option>
                <Option value="post_call">post_call</Option>
              </>
            )}
          </Select>
        </Form.Item>

        <Form.Item
          name="default_on"
          label="Always On"
          tooltip="If enabled, this guardrail will be applied to all requests by default"
          valuePropName="checked"
        >
          <Switch />
        </Form.Item>

        <Form.Item
          name="skip_system_message_choice"
          label="Skip system messages in guardrail"
          tooltip="Unified guardrails only: whether role: system content is omitted from guardrail input (LLM still receives full messages). Use global default follows litellm_settings.skip_system_message_in_guardrail."
        >
          <Select>
            <Option value="inherit">Use global default</Option>
            <Option value="yes">Yes — exclude from guardrail scan</Option>
            <Option value="no">No — always include in scan</Option>
          </Select>
        </Form.Item>

        {renderProviderSpecificFields()}

        <div className="flex justify-end space-x-2 mt-4">
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} loading={loading}>
            Update Guardrail
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default EditGuardrailForm;
