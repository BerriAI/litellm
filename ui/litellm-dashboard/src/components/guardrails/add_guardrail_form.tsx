import React, { useState, useEffect } from 'react';
import { Card, Form, Typography, Select, Input, Switch, Tooltip, Modal, message, Divider, Space, Tag, Image, Steps } from 'antd';
import { Button, TextInput } from '@tremor/react';
import type { FormInstance } from 'antd';
import { GuardrailProviders, guardrail_provider_map, provider_specific_fields, guardrailLogoMap } from './guardrail_info_helpers';
import { createGuardrailCall, getGuardrailUISettings } from '../networking';
import PiiConfiguration from './pii_configuration';

const { Title, Text, Link } = Typography;
const { Option } = Select;
const { Step } = Steps;

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

const AddGuardrailForm: React.FC<AddGuardrailFormProps> = ({ 
  visible, 
  onClose, 
  accessToken,
  onSuccess
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [guardrailSettings, setGuardrailSettings] = useState<GuardrailSettings | null>(null);
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [selectedActions, setSelectedActions] = useState<{[key: string]: string}>({});
  const [currentStep, setCurrentStep] = useState(0);

  // Fetch guardrail settings when the component mounts
  useEffect(() => {
    const fetchGuardrailSettings = async () => {
      try {
        if (!accessToken) return;
        
        const data = await getGuardrailUISettings(accessToken);
        setGuardrailSettings(data);
      } catch (error) {
        console.error('Error fetching guardrail settings:', error);
        message.error('Failed to load guardrail settings');
      }
    };
    
    fetchGuardrailSettings();
  }, [accessToken]);

  const handleProviderChange = (value: string) => {
    setSelectedProvider(value);
    // Reset form fields that are provider-specific
    form.setFieldsValue({
      config: undefined
    });
    
    // Reset PII selections when changing provider
    setSelectedEntities([]);
    setSelectedActions({});
  };

  const handleEntitySelect = (entity: string) => {
    setSelectedEntities(prev => {
      if (prev.includes(entity)) {
        return prev.filter(e => e !== entity);
      } else {
        return [...prev, entity];
      }
    });
  };

  const handleActionSelect = (entity: string, action: string) => {
    setSelectedActions(prev => ({
      ...prev,
      [entity]: action
    }));
  };

  const nextStep = async () => {
    try {
      // Validate current step fields
      if (currentStep === 0) {
        await form.validateFields(['guardrail_name', 'provider', 'mode', 'default_on']);
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
    setCurrentStep(0);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      // First validate currently visible fields (step 1)
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
          default_on: values.default_on
        },
        guardrail_info: {}
      };

      // For Presidio PII, add the entity and action configurations
      if (values.provider === 'PresidioPII' && selectedEntities.length > 0) {
        const piiEntitiesConfig: {[key: string]: string} = {};
        selectedEntities.forEach(entity => {
          piiEntitiesConfig[entity] = selectedActions[entity] || 'MASK'; // Default to MASK if no action selected
        });
        
        guardrailData.litellm_params.pii_entities_config = piiEntitiesConfig;
      }
      // Add config values to the guardrail_info if provided
      else if (values.config) {
        try {
          const configObj = JSON.parse(values.config);
          // For some guardrails, the config values need to be in litellm_params
          // Especially for providers like Bedrock that need guardrailIdentifier and guardrailVersion
          if (values.provider === 'Bedrock' && configObj) {
            if (configObj.guardrail_id) {
              guardrailData.litellm_params.guardrailIdentifier = configObj.guardrail_id;
            }
            if (configObj.guardrail_version) {
              guardrailData.litellm_params.guardrailVersion = configObj.guardrail_version;
            }
          } else {
            // For other providers, add the config to guardrail_info
            guardrailData.guardrail_info = configObj;
          }
        } catch (error) {
          message.error('Invalid JSON in configuration');
          setLoading(false);
          return;
        }
      }

      if (!accessToken) {
        throw new Error("No access token available");
      }

      console.log("Sending guardrail data:", JSON.stringify(guardrailData));
      await createGuardrailCall(accessToken, guardrailData);
      
      message.success('Guardrail created successfully');
      
      // Reset form and close modal
      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create guardrail:", error);
      message.error('Failed to create guardrail: ' + (error instanceof Error ? error.message : String(error)));
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
          rules={[{ required: true, message: 'Please enter a guardrail name' }]}
        >
          <TextInput placeholder="Enter a name for this guardrail" />
        </Form.Item>

        <Form.Item
          name="provider"
          label="Guardrail Provider"
          rules={[{ required: true, message: 'Please select a provider' }]}
        >
          <Select 
            placeholder="Select a guardrail provider"
            onChange={handleProviderChange}
            optionLabelProp="label"
          >
            {Object.entries(GuardrailProviders).map(([key, value]) => (
              <Option 
                key={key} 
                value={key}
                label={value}
              >
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  {guardrailLogoMap[value] && (
                    <img 
                      src={guardrailLogoMap[value]} 
                      alt=""
                      style={{ 
                        height: '20px', 
                        width: '20px', 
                        marginRight: '8px',
                        objectFit: 'contain'
                      }}
                      onError={(e) => {
                        // Hide broken image icon if image fails to load
                        e.currentTarget.style.display = 'none';
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
          rules={[{ required: true, message: 'Please select a mode' }]}
        >
          <Select>
            {guardrailSettings?.supported_modes?.map(mode => (
              <Option key={mode} value={mode}>{mode}</Option>
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
      </>
    );
  };

  const renderPiiConfiguration = () => {
    if (!guardrailSettings || selectedProvider !== 'PresidioPII') return null;
    
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

  const renderProviderSpecificConfig = () => {
    if (!selectedProvider || selectedProvider === 'PresidioPII') return null;

    switch (selectedProvider) {
      case 'Aporia':
        return (
          <Form.Item
            label="Aporia Configuration"
            name="config"
            tooltip="JSON configuration for Aporia"
          >
            <Input.TextArea
              rows={4}
              placeholder={`{
  "api_key": "your_aporia_api_key",
  "project_name": "your_project_name"
}`}
            />
          </Form.Item>
        );
      case 'AimSecurity':
        return (
          <Form.Item
            label="Aim Security Configuration"
            name="config"
            tooltip="JSON configuration for Aim Security"
          >
            <Input.TextArea
              rows={4}
              placeholder={`{
  "api_key": "your_aim_api_key"
}`}
            />
          </Form.Item>
        );
      case 'Bedrock':
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
      case 'GuardrailsAI':
        return (
          <Form.Item
            label="Guardrails.ai Configuration"
            name="config"
            tooltip="JSON configuration for Guardrails.ai"
          >
            <Input.TextArea
              rows={4}
              placeholder={`{
  "api_key": "your_guardrails_api_key",
  "guardrail_id": "your_guardrail_id"
}`}
            />
          </Form.Item>
        );
      case 'LakeraAI':
        return (
          <Form.Item
            label="Lakera AI Configuration"
            name="config"
            tooltip="JSON configuration for Lakera AI"
          >
            <Input.TextArea
              rows={4}
              placeholder={`{
  "api_key": "your_lakera_api_key"
}`}
            />
          </Form.Item>
        );
      case 'PromptInjection':
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
          <Form.Item
            label="Custom Configuration"
            name="config"
            tooltip="JSON configuration for your custom guardrail"
          >
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

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return renderBasicInfo();
      case 1:
        if (selectedProvider === 'PresidioPII') {
          return renderPiiConfiguration();
        } else {
          return renderProviderSpecificConfig();
        }
      default:
        return null;
    }
  };

  const renderStepButtons = () => {
    return (
      <div className="flex justify-end space-x-2 mt-4">
        {currentStep > 0 && (
          <Button 
            variant="secondary"
            onClick={prevStep}
          >
            Previous
          </Button>
        )}
        {currentStep < 1 && (
          <Button 
            onClick={nextStep}
          >
            Next
          </Button>
        )}
        {currentStep === 1 && (
          <Button 
            onClick={handleSubmit}
            loading={loading}
          >
            Create Guardrail
          </Button>
        )}
        <Button 
          variant="secondary"
          onClick={handleClose}
        >
          Cancel
        </Button>
      </div>
    );
  };

  return (
    <Modal
      title="Add Guardrail"
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={700}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          mode: guardrailSettings?.supported_modes?.[0] || "pre_call",
          default_on: false
        }}
      >
        <Steps current={currentStep} className="mb-6">
          <Step title="Basic Info" />
          <Step title={selectedProvider === 'PresidioPII' ? "PII Configuration" : "Provider Configuration"} />
        </Steps>
        
        {renderStepContent()}
        {renderStepButtons()}
      </Form>
    </Modal>
  );
};

export default AddGuardrailForm; 