import React, { useState } from 'react';
import { Card, Form, Typography, Select, Input, Switch, Tooltip, Modal } from 'antd';
import { Button } from '@tremor/react';
import type { FormInstance } from 'antd';
import { GuardrailProviders, guardrail_provider_map } from './guardrail_info_helpers';
import { createGuardrailCall } from '../networking';

const { Title, Text, Link } = Typography;
const { Option } = Select;

interface AddGuardrailFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
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

  const handleProviderChange = (value: string) => {
    setSelectedProvider(value);
    // Reset form fields that are provider-specific
    form.setFieldsValue({
      config: undefined
    });
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();
      
      // Prepare the guardrail data
      const guardrailData = {
        guardrail_name: values.guardrail_name,
        litellm_params: {
          guardrail: guardrail_provider_map[values.provider],
          mode: values.mode,
          default_on: values.default_on
        },
        guardrail_info: {
          // Parse the config if it's provided as JSON
          ...(values.config ? JSON.parse(values.config) : {})
        }
      };

      if (!accessToken) {
        throw new Error("No access token available");
      }

      await createGuardrailCall(accessToken, guardrailData);
      
      // Reset form and close modal
      form.resetFields();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create guardrail:", error);
    } finally {
      setLoading(false);
    }
  };

  const renderProviderSpecificFields = () => {
    if (!selectedProvider) return null;

    switch (selectedProvider) {
      case 'PresidioPII':
        return (
          <Form.Item
            label="PII Configuration"
            name="config"
            tooltip="JSON configuration for Presidio PII detection"
          >
            <Input.TextArea
              rows={4}
              placeholder={`{
  "entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
  "language": "en"
}`}
            />
          </Form.Item>
        );
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

  return (
    <Modal
      title="Add Guardrail"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={700}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          mode: "embedded",
          default_on: false
        }}
      >
        <Form.Item
          name="guardrail_name"
          label="Guardrail Name"
          rules={[{ required: true, message: 'Please enter a guardrail name' }]}
        >
          <Input placeholder="Enter a name for this guardrail" />
        </Form.Item>

        <Form.Item
          name="provider"
          label="Guardrail Provider"
          rules={[{ required: true, message: 'Please select a provider' }]}
        >
          <Select 
            placeholder="Select a guardrail provider"
            onChange={handleProviderChange}
          >
            {Object.entries(GuardrailProviders).map(([key, value]) => (
              <Option key={key} value={key}>
                {value}
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
            <Option value="embedded">Embedded</Option>
            <Option value="proxy">Proxy</Option>
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

        {renderProviderSpecificFields()}

        <div className="flex justify-end space-x-2 mt-4">
          <Button onClick={onClose}>Cancel</Button>
          <Button 
            onClick={handleSubmit}
            loading={loading}
          >
            Create Guardrail
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default AddGuardrailForm; 