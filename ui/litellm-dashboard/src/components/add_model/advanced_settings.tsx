import React, { useState } from "react";
import { Form, Switch, Input, Select } from "antd";
import { Text, TextInput, Button, Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
import { Row, Col, Typography, Card } from "antd";
import TextArea from "antd/es/input/TextArea";
const { Link } = Typography;

interface AdvancedSettingsProps {
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
}

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({
  showAdvancedSettings,
  setShowAdvancedSettings,
}) => {
  const [form] = Form.useForm();
  const [pricingModel, setPricingModel] = useState<'per_second' | 'per_token'>('per_token');
  const [showCustomPricing, setShowCustomPricing] = useState(false);

  // Add validation function
  const validateJSON = (_: any, value: string) => {
    if (!value) {
      return Promise.resolve();
    }
    try {
      JSON.parse(value);
      return Promise.resolve();
    } catch (error) {
      return Promise.reject('Please enter valid JSON');
    }
  };

  const validateNumber = (_: any, value: string) => {
    if (!value) {
      return Promise.resolve();
    }
    if (isNaN(Number(value)) || Number(value) < 0) {
      return Promise.reject('Please enter a valid positive number');
    }
    return Promise.resolve();
  };

  const handlePassThroughChange = (checked: boolean) => {
    const currentParams = form.getFieldValue('litellm_extra_params');
    try {
      let paramsObj = currentParams ? JSON.parse(currentParams) : {};
      if (checked) {
        paramsObj.use_in_pass_through = true;
      } else {
        delete paramsObj.use_in_pass_through;
      }
      // Only set the field value if there are remaining parameters
      if (Object.keys(paramsObj).length > 0) {
        form.setFieldValue('litellm_extra_params', JSON.stringify(paramsObj, null, 2));
      } else {
        form.setFieldValue('litellm_extra_params', '');
      }
    } catch (error) {
      // If JSON parsing fails, only create new object if checked is true
      if (checked) {
        form.setFieldValue('litellm_extra_params', JSON.stringify({ use_in_pass_through: true }, null, 2));
      } else {
        form.setFieldValue('litellm_extra_params', '');
      }
    }
  };

  const updateLiteLLMParams = (updates: any) => {
    const currentParams = form.getFieldValue('litellm_extra_params');
    try {
      let paramsObj = currentParams ? JSON.parse(currentParams) : {};
      paramsObj = { ...paramsObj, ...updates };
      
      // Remove any null or undefined values
      Object.keys(paramsObj).forEach(key => {
        if (paramsObj[key] === null || paramsObj[key] === undefined) {
          delete paramsObj[key];
        }
      });

      // Only set the field value if there are parameters
      if (Object.keys(paramsObj).length > 0) {
        form.setFieldValue('litellm_extra_params', JSON.stringify(paramsObj, null, 2));
      } else {
        form.setFieldValue('litellm_extra_params', '');
      }
    } catch (error) {
      // If JSON parsing fails, create new object with updates
      form.setFieldValue('litellm_extra_params', JSON.stringify(updates, null, 2));
    }
  };

  const handlePricingChange = (value: string | number | null, type: string) => {
    if (value === '' || value === null) {
      return;
    }

    const numValue = Number(value);
    if (isNaN(numValue)) {
      return;
    }

    let updates: any = {};
    
    if (pricingModel === 'per_second') {
      updates.input_cost_per_second = numValue;
    } else {
      // Convert from per million tokens to per token
      const perTokenValue = numValue / 1_000_000;
      if (type === 'input') {
        updates.input_cost_per_token = perTokenValue;
      } else {
        updates.output_cost_per_token = perTokenValue;
      }
    }

    updateLiteLLMParams(updates);
  };

  const handlePricingModelChange = (value: 'per_token' | 'per_second') => {
    setPricingModel(value);
    // Clear existing pricing in litellm_params when switching models
    const currentParams = form.getFieldValue('litellm_extra_params');
    try {
      let paramsObj = currentParams ? JSON.parse(currentParams) : {};
      delete paramsObj.input_cost_per_second;
      delete paramsObj.input_cost_per_token;
      delete paramsObj.output_cost_per_token;
      form.setFieldValue('litellm_extra_params', 
        Object.keys(paramsObj).length > 0 ? JSON.stringify(paramsObj, null, 2) : ''
      );
    } catch (error) {
      form.setFieldValue('litellm_extra_params', '');
    }
    // Clear the form fields
    form.setFieldsValue({
      input_cost_per_second: undefined,
      input_cost_per_million_tokens: undefined,
      output_cost_per_million_tokens: undefined
    });
  };

  return (
    <>
      <Accordion className="mt-2 mb-4">
        <AccordionHeader>
          <b>Advanced Settings</b>
        </AccordionHeader>
        <AccordionBody>
          <div className="bg-white rounded-lg">
            <Form.Item
              label="Custom Pricing"
              name="custom_pricing_enabled"
              valuePropName="checked"
              className="mb-4"
              tooltip={
                <span>
                  LiteLLM tracks pricing for all models by default. View default model prices{" "}
                  <Link href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json" target="_blank">
                    here
                  </Link>. Only set this if you want to use custom pricing for this model.
                </span>
              }
            >
              <Switch
                onChange={setShowCustomPricing}
                className="bg-gray-600"
              />
            </Form.Item>

            {showCustomPricing && (
              <div className="ml-6 border-l-2 pl-4 border-gray-200 mb-10">
                <Form.Item
                  label="Pricing Model"
                  name="pricing_model"
                  className="mb-4"
                  tooltip="Select pricing model type"
                  initialValue="per_token"
                >
                  <Select
                    onChange={handlePricingModelChange}
                    options={[
                      { value: 'per_token', label: 'Per Token' },
                      { value: 'per_second', label: 'Per Second' },
                    ]}
                  />
                </Form.Item>

                {pricingModel === 'per_second' ? (
                  <Form.Item
                    label="Cost Per Second"
                    name="input_cost_per_second"
                    tooltip="Cost per second of model usage"
                    rules={[{ validator: validateNumber }]}
                    className="mb-4"
                  >
                    <TextInput 
                      placeholder="0.0001" 
                      onChange={(e) => handlePricingChange(e.target.value, 'input')}
                    />
                  </Form.Item>
                ) : (
                  <>
                    <Form.Item
                      label="Input Cost (per 1M tokens)"
                      name="input_cost_per_million_tokens"
                      tooltip="Cost per 1 million input tokens"
                      rules={[{ validator: validateNumber }]}
                      className="mb-4"
                    >
                      <TextInput 
                        placeholder="1.00" 
                        onChange={(e) => handlePricingChange(e.target.value, 'input')}
                      />
                    </Form.Item>
                    <Form.Item
                      label="Output Cost (per 1M tokens)"
                      name="output_cost_per_million_tokens"
                      tooltip="Cost per 1 million output tokens"
                      rules={[{ validator: validateNumber }]}
                      className="mb-4"
                    >
                      <TextInput 
                        placeholder="2.00" 
                        onChange={(e) => handlePricingChange(e.target.value, 'output')}
                      />
                    </Form.Item>
                  </>
                )}
              </div>
            )}

            <Form.Item
              label="Use in pass through routes"
              name="use_in_pass_through"
              valuePropName="checked"
              className="mb-4 mt-4"
              tooltip={
                <span>
                  Allow using these credentials in pass through routes.{" "}
                  <Link href="https://docs.litellm.ai/docs/pass_through/vertex_ai" target="_blank">
                    Learn more
                  </Link>
                </span>
              }
            >
              <Switch 
                onChange={handlePassThroughChange} 
                className="bg-gray-600" 
              />
            </Form.Item>

            <Form.Item
              label="LiteLLM Params"
              name="litellm_extra_params"
              tooltip="Optional litellm params used for making a litellm.completion() call."
              className="mb-4 mt-4"
              rules={[{ validator: validateJSON }]}
            >
              <TextArea
                rows={4}
                placeholder='{
                  "rpm": 100,
                  "timeout": 0,
                  "stream_timeout": 0
                }'
              />
            </Form.Item>
            <Row className="mb-4">
              <Col span={10}></Col>
              <Col span={10}>
                <Text className="text-gray-600 text-sm">
                  Pass JSON of litellm supported params{" "}
                  <Link
                    href="https://docs.litellm.ai/docs/completion/input"
                    target="_blank"
                  >
                    litellm.completion() call
                  </Link>
                </Text>
              </Col>
            </Row>
            <Form.Item
              label="Model Info"
              name="model_info_params"
              tooltip="Optional model info params. Returned when calling `/model/info` endpoint."
              className="mb-0"
              rules={[{ validator: validateJSON }]}
            >
              <TextArea
                rows={4}
                placeholder='{
                  "mode": "chat"
                }'
              />
            </Form.Item>
          </div>
        </AccordionBody>
      </Accordion>
    </>
  );
};

export default AdvancedSettings;