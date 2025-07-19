import React, { useState } from "react";
import {
  TextInput,
  Icon,
  Button as TremorButton,
  Text,
} from "@tremor/react";
import {
  Modal,
  Form,
  Select,
  message,
  Tooltip,
  Input,
  Alert,
} from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { CredentialItem, vectorStoreCreateCall } from "../networking";
import { VectorStoreProviders, vectorStoreProviderLogoMap, vectorStoreProviderMap, getProviderSpecificFields, VectorStoreFieldConfig } from "../vector_store_providers";

interface VectorStoreFormProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  credentials: CredentialItem[];
}

const VectorStoreForm: React.FC<VectorStoreFormProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  accessToken,
  credentials,
}) => {
  const [form] = Form.useForm();
  const [metadataJson, setMetadataJson] = useState("{}");
  const [selectedProvider, setSelectedProvider] = useState("bedrock");

  const handleCreate = async (formValues: any) => {
    if (!accessToken) return;
    try {
      // Parse metadata JSON
      let metadata = {};
      try {
        metadata = metadataJson.trim() ? JSON.parse(metadataJson) : {};
      } catch (e) {
        message.error("Invalid JSON in metadata field");
        return;
      }

      // Prepare the payload with provider-specific fields
      const payload: any = {
        vector_store_id: formValues.vector_store_id,
        custom_llm_provider: formValues.custom_llm_provider,
        vector_store_name: formValues.vector_store_name,
        vector_store_description: formValues.vector_store_description,
        vector_store_metadata: metadata,
        litellm_credential_name: formValues.litellm_credential_name,
      };

      // pass all provider fields as litellm params dict
      const providerFields = getProviderSpecificFields(formValues.custom_llm_provider);
      const litellmParams = providerFields.reduce((acc, field) => {
        acc[field.name] = formValues[field.name];
        return acc;
      }, {} as Record<string, any>);

      payload["litellm_params"] = litellmParams;

      await vectorStoreCreateCall(accessToken, payload);
      message.success("Vector store created successfully");
      form.resetFields();
      setMetadataJson("{}");
      onSuccess();
    } catch (error) {
      console.error("Error creating vector store:", error);
      message.error("Error creating vector store: " + error);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    setMetadataJson("{}");
    setSelectedProvider("bedrock");
    onCancel();
  };

  return (
    <Modal
      title="Add New Vector Store"
      visible={isVisible}
      width={1000}
      footer={null}
      onCancel={handleCancel}
    >
      <Form
        form={form}
        onFinish={handleCreate}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
      >
        <Form.Item
          label={
            <span>
              Provider{' '}
              <Tooltip title="Select the provider for this vector store">
                <InfoCircleOutlined style={{ marginLeft: '4px' }} />
              </Tooltip>
            </span>
          }
          name="custom_llm_provider"
          rules={[{ required: true, message: "Please select a provider" }]}
          initialValue="bedrock"
        >
          <Select onChange={(value) => setSelectedProvider(value)}>
            {Object.entries(VectorStoreProviders).map(([providerEnum, providerDisplayName]) => {
              return (
                <Select.Option key={providerEnum} value={vectorStoreProviderMap[providerEnum]}>
                  <div className="flex items-center space-x-2">
                    <img
                      src={vectorStoreProviderLogoMap[providerDisplayName]}
                      alt={`${providerEnum} logo`}
                      className="w-5 h-5"
                      onError={(e) => {
                        // Create a div with provider initial as fallback
                        const target = e.target as HTMLImageElement;
                        const parent = target.parentElement;
                        if (parent) {
                          const fallbackDiv = document.createElement('div');
                          fallbackDiv.className = 'w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs';
                          fallbackDiv.textContent = providerDisplayName.charAt(0);
                          parent.replaceChild(fallbackDiv, target);
                        }
                      }}
                    />
                    <span>{providerDisplayName}</span>
                  </div>
                </Select.Option>
              );
            })}
          </Select>
        </Form.Item>

        {/* PG Vector Setup Instructions */}
        {selectedProvider === "pg_vector" && (
          <Alert
            message="PG Vector Setup Required"
            description={
              <div>
                <p>LiteLLM provides a server to connect to PG Vector. To use this provider:</p>
                <ol style={{ marginLeft: '16px', marginTop: '8px' }}>
                  <li>Deploy the litellm-pgvector server from: <a href="https://github.com/BerriAI/litellm-pgvector" target="_blank" rel="noopener noreferrer">https://github.com/BerriAI/litellm-pgvector</a></li>
                  <li>Configure your PostgreSQL database with pgvector extension</li>
                  <li>Start the server and note the API base URL and API key</li>
                  <li>Enter those details in the fields below</li>
                </ol>
              </div>
            }
            type="info"
            showIcon
            style={{ marginBottom: '16px' }}
          />
        )}

        <Form.Item
          label={
              <span>
                Vector Store ID{' '}
                <Tooltip title="Enter the vector store ID from your api provider">
                  <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                </Tooltip>
              </span>
          }
          name="vector_store_id"
          rules={[{ required: true, message: "Please input the vector store ID from your api provider" }]}
        >
          <TextInput />
        </Form.Item>

        {/* Provider-specific fields */}
        {getProviderSpecificFields(selectedProvider).map((field: VectorStoreFieldConfig) => (
          <Form.Item
            key={field.name}
            label={
              <span>
                {field.label}{' '}
                <Tooltip title={field.tooltip}>
                  <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                </Tooltip>
              </span>
            }
            name={field.name}
            rules={field.required ? [{ required: true, message: `Please input the ${field.label.toLowerCase()}` }] : []}
          >
            <TextInput 
              type={field.type || "text"}
              placeholder={field.placeholder} 
            />
          </Form.Item>
        ))}

        <Form.Item
          label={
              <span>
                  Vector Store Name{' '}
                  <Tooltip title="Custom name you want to give to the vector store, this name will be rendered on the LiteLLM UI">
                      <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                  </Tooltip>
              </span>
          }
          name="vector_store_name"
        >
          <TextInput />
        </Form.Item>

        <Form.Item
          label="Description"
          name="vector_store_description"
        >
          <Input.TextArea rows={4} />
        </Form.Item>

        <Form.Item
          label={
              <span>
                  Existing Credentials{' '}
                  <Tooltip title="Optionally select API provider credentials for this vector store eg. Bedrock API KEY">
                      <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                  </Tooltip>
              </span>
          }
          name="litellm_credential_name"
        >
          <Select
            showSearch
            placeholder="Select or search for existing credentials"
            optionFilterProp="children"
            filterOption={(input, option) =>
              (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
            }
            options={[
              { value: null, label: 'None' },
              ...credentials.map((credential) => ({
                value: credential.credential_name,
                label: credential.credential_name
              }))
            ]}
            allowClear
          />
        </Form.Item>

        <Form.Item
          label={
            <span>
              Metadata{' '}
              <Tooltip title="JSON metadata for the vector store (optional)">
                <InfoCircleOutlined style={{ marginLeft: '4px' }} />
              </Tooltip>
            </span>
          }
        >
          <Input.TextArea
            rows={4}
            value={metadataJson}
            onChange={(e) => setMetadataJson(e.target.value)}
            placeholder='{"key": "value"}'
          />
        </Form.Item>

        <div className="flex justify-end space-x-3">
          <TremorButton onClick={handleCancel} variant="secondary">
            Cancel
          </TremorButton>
          <TremorButton variant="primary" type="submit">
            Create
          </TremorButton>
        </div>
      </Form>
    </Modal>
  );
};

export default VectorStoreForm; 