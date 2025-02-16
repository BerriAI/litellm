import React from "react";
import { Form, Select as AntSelect } from "antd";
import { TextInput, Text } from "@tremor/react";
import { Row, Col } from "antd";
import { Providers, provider_map } from "../provider_info_helpers";

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

  const handleModelChange = (value: string) => {
    console.log("VALUE:", "|", value, value == 'all-wildcard',"|", selectedProvider);
    if (value == 'all-wildcard' && selectedProvider) {
      // Get the litellm provider name from provider_map
      const litellmProvider = provider_map[selectedProvider]?.toLowerCase();
      if (litellmProvider) {
        const providerLiteLLMName = `${litellmProvider}/*`;
        form.setFieldsValue({ 
          model: providerLiteLLMName,
          model_name: providerLiteLLMName 
        });
      }
    } else if (value === 'custom') {
      form.setFieldsValue({ 
        model: value,
        model_name: "" 
      });
    } else if (value && selectedProvider) {
      // For specific model selection, prefix with provider name
      const litellmProvider = provider_map[selectedProvider]?.toLowerCase();
      if (litellmProvider) {
        const modelName = `${litellmProvider}/${value}`;
        form.setFieldsValue({ 
          model: modelName,
          model_name: modelName 
        });
      }
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
            const model = getFieldValue('model') || "";
            return model === 'custom' && (
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