import React from "react";
import { Form, Select as AntSelect } from "antd";
import { TextInput, Text } from "@tremor/react";
import { Row, Col } from "antd";
import { Providers } from "../provider_info_helpers";

interface LiteLLMModelNameFieldProps {
  selectedProvider: string;
  providerModels: string[];
  getPlaceholder: (provider: string) => string;
}

const LiteLLMModelNameField: React.FC<LiteLLMModelNameFieldProps> = ({
  selectedProvider,
  providerModels,
  getPlaceholder,
}) => {
  const form = Form.useFormInstance();

  const handleModelChange = (value: string[]) => {
    // If "all-wildcard" is selected, clear the model_name field
    if (value.includes("all-wildcard")) {
      form.setFieldsValue({ model_name: undefined });
    }
  };

  return (
    <>
      <Form.Item
        label="LiteLLM Model Name(s)"
        tooltip="Actual model name used for making litellm.completion() / litellm.embedding() call."
        className="mb-0"
      >
        <Form.Item
          name="model"
          rules={[{ required: true, message: "Please select at least one model." }]}
          noStyle
        >
          {(selectedProvider === Providers.Azure) || 
           (selectedProvider === Providers.OpenAI_Compatible) || 
           (selectedProvider === Providers.Ollama) ? (
            <TextInput placeholder={getPlaceholder(selectedProvider.toString())} />
          ) : providerModels.length > 0 ? (
            <AntSelect
              allowClear
              showSearch
              placeholder="Select models"
              onChange={handleModelChange}
              optionFilterProp="children"
              filterOption={(input, option) =>
                (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
              }
              options={[
                {
                  label: `All ${selectedProvider} Models (Wildcard)`,
                  value: 'all-wildcard'
                },
                ...providerModels.map(model => ({
                  label: model,
                  value: model
                })),
                {
                  label: 'Custom Model Name (Enter below)',
                  value: 'custom'
                }
              ]}
              style={{ width: '100%' }}
            />
          ) : (
            <TextInput placeholder={getPlaceholder(selectedProvider.toString())} />
          )}
        </Form.Item>

        {/* Custom Model Name field */}
        <Form.Item
          noStyle
          shouldUpdate={(prevValues, currentValues) => 
            prevValues.model !== currentValues.model
          }
        >
          {({ getFieldValue }) => {
            const selectedModels = getFieldValue('model') || [];
            return selectedModels.includes('custom') && (
              <Form.Item
                name="custom_model_name"
                rules={[{ required: true, message: "Please enter a custom model name." }]}
                className="mt-2"
              >
                <TextInput placeholder="Enter custom model name" />
              </Form.Item>
            );
          }}
        </Form.Item>
      </Form.Item>
      <Row>
        <Col span={10}></Col>
        <Col span={10}>
          <Text className="mb-3 mt-1">
            Actual model name used for making litellm.completion() call. We loadbalance models with the same public name
          </Text>
        </Col>
      </Row>
    </>
  );
};

export default LiteLLMModelNameField;