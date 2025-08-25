import React from "react";
import { Form, Switch, Select, Input, Tooltip, Table, InputNumber } from "antd";
import { Text, Button, Accordion, AccordionHeader, AccordionBody, TextInput } from "@tremor/react";
import { Row, Col, Typography } from "antd";
import TextArea from "antd/es/input/TextArea";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Team } from "../key_team_helpers/key_list";
import TeamDropdown from "../common_components/team_dropdown";
import CacheControlSettings from "./cache_control_settings";

const { Link } = Typography;

interface AdvancedSettingsProps {
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams?: Team[] | null;
  guardrailsList: string[];
}

interface WeightExample {
  key: string;
  public_model_name: string;
  litellm_model_name: string;
  weight: number;
  description: string;
  requests_per_100: string;
}

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({
  showAdvancedSettings,
  setShowAdvancedSettings,
  teams,
  guardrailsList,
}) => {
  const [form] = Form.useForm();
    const [customPricing, setCustomPricing] = React.useState(false);
  const [pricingModel, setPricingModel] = React.useState<'per_token' | 'per_second'>('per_token');
  const [showCacheControl, setShowCacheControl] = React.useState(false);
  
  // Weight management states
  const [weightExamples, setWeightExamples] = React.useState<WeightExample[]>([
    {
      key: '1',
      public_model_name: 'gpt-4',
      litellm_model_name: 'openai/gpt-4',
      weight: 2,
      description: 'Default selection probability',
      requests_per_100: '~50%'
    },
    {
      key: '2',
      public_model_name: 'gpt-4-turbo',
      litellm_model_name: 'openai/gpt-4-turbo',
      weight: 1,
      description: 'Selected 2x more often',
      requests_per_100: '~25%'
    },
    {
      key: '3',
      public_model_name: 'gpt-4-preview',
      litellm_model_name: 'openai/gpt-4-preview',
      weight: 1,
      description: 'Selected half as often',
      requests_per_100: '~25%'
    }
  ]);
  
  // Consistent styling for all description text
  const descriptionTextStyle = "text-sm text-gray-600";
  
  // Add validation function for numbers
  const validateNumber = (_: any, value: string) => {
    if (!value) {
      return Promise.resolve();
    }
    if (isNaN(Number(value)) || Number(value) < 0) {
      return Promise.reject('Please enter a valid positive number');
    }
    return Promise.resolve();
  };

  // Calculate requests per 100
  const calculateRequestsPer100 = (weight: number, examples: WeightExample[]): string => {
    const totalWeight = examples.reduce((sum, example) => sum + example.weight, 0);
    const requestsPer100 = Math.round((weight / totalWeight) * 100);
    return `~${requestsPer100}%`;
  };

  // Handle individual weight changes
  const handleIndividualWeightChange = (key: string, newWeight: number) => {
    // First, update the weight of the changed model
    const updatedExamples = weightExamples.map(example => {
      if (example.key === key) {
        return {
          ...example,
          weight: newWeight
        };
      }
      return example;
    });
    
    // Then recalculate requests per 100 for ALL models
    const finalExamples = updatedExamples.map(example => {
      const newRequestsPer100 = calculateRequestsPer100(example.weight, updatedExamples);
      return {
        ...example,
        requests_per_100: newRequestsPer100
      };
    });
    
    setWeightExamples(finalExamples);
  };

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

  // Handle custom pricing changes
  const handleCustomPricingChange = (checked: boolean) => {
    setCustomPricing(checked);
    if (!checked) {
      // Clear pricing fields when disabled
      form.setFieldsValue({
        input_cost_per_token: undefined,
        output_cost_per_token: undefined,
        input_cost_per_second: undefined,
      });
    }
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

  const handleCacheControlChange = (checked: boolean) => {
    setShowCacheControl(checked);
    if (!checked) {
      const currentParams = form.getFieldValue('litellm_extra_params');
      try {
        let paramsObj = currentParams ? JSON.parse(currentParams) : {};
        delete paramsObj.cache_control_injection_points;
        if (Object.keys(paramsObj).length > 0) {
          form.setFieldValue('litellm_extra_params', JSON.stringify(paramsObj, null, 2));
        } else {
          form.setFieldValue('litellm_extra_params', '');
        }
      } catch (error) {
        form.setFieldValue('litellm_extra_params', '');
      }
    }
  };



  const weightColumns = [
    {
      title: 'Public Model (Custom Name)',
      dataIndex: 'public_model_name',
      key: 'public_model_name',
      width: '25%',
      render: (text: string) => (
        <div className="text-sm font-medium text-gray-900">{text}</div>
      )
    },
    {
      title: 'LiteLLM Model Name',
      dataIndex: 'litellm_model_name',
      key: 'litellm_model_name',
      width: '25%',
      render: (text: string) => (
        <div className="text-sm text-gray-700">{text}</div>
      )
    },
    {
      title: 'Weight',
      dataIndex: 'weight',
      key: 'weight',
      width: '20%',
      render: (weight: number, record: WeightExample) => (
        <InputNumber
          value={weight}
          min={0}
          step={0.1}
          size="small"
          className="w-full"
          onChange={(value) => handleIndividualWeightChange(record.key, value || 0)}
        />
      )
    },
    {
      title: 'Requests handled by Model',
      dataIndex: 'requests_per_100',
      key: 'requests_per_100',
      width: '30%',
      render: (text: string) => (
        <div className="text-sm font-medium text-green-600">{text}</div>
      )
    }
  ];

  return (
    <>
      <Accordion className="mt-2 mb-4 border-gray-300">
        <AccordionHeader>
          <h4 className="text-lg font-semibold text-gray-900">Advanced Settings</h4>
        </AccordionHeader>
        <AccordionBody>
          <div className="bg-white rounded-lg">

            {/* Weight Section */}
            <div className="border border-gray-200 rounded-lg p-4 mb-6">
              <div className="mb-4">
                <h5 className="text-base font-semibold text-gray-900 mb-1">Manage Model Weights</h5>
                <p className={descriptionTextStyle}>Configure how often this model deployment is selected relative to others with the same model name</p>
              </div>

              {/* Weight Examples Table */}
              <div className="mt-6">
                <h6 className="text-sm font-semibold text-gray-700 mb-3">Models in your configuration</h6>
                <Table 
                  dataSource={weightExamples} 
                  columns={weightColumns} 
                  pagination={false}
                  size="small"
                  className="weight-examples-table border border-gray-200 rounded-lg overflow-hidden"
                  rowClassName={(record, index) => index % 2 === 0 ? '' : 'bg-gray-50'}
                />
                <div className="mt-3 text-xs text-gray-500">
                  <strong>Note:</strong> These percentages are approximate and may vary based on the total number of models and their weights in your configuration. You can edit individual weights in the table above.
                </div>
              </div>
            </div>

            {/* Pricing Section */}
            <div className="border border-gray-200 rounded-lg p-4 mb-6">
              <div className="mb-4">
                <h5 className="text-base font-semibold text-gray-900 mb-1">Custom Pricing</h5>
                <p className={descriptionTextStyle}>Override default pricing for cost tracking and billing</p>
              </div>

              <Form.Item
                label="Enable Custom Pricing"
                name="custom_pricing"
                valuePropName="checked"
                className="mb-4"
              >
                <Switch onChange={handleCustomPricingChange} className="bg-gray-600" />
              </Form.Item>


            {customPricing && (
              <div className="ml-6 pl-4 border-l-2 border-gray-200">
                <Form.Item
                  label="Pricing Model"
                  name="pricing_model"
                  className="mb-4"
                >
                  <Select
                    defaultValue="per_token"
                    onChange={(value: 'per_token' | 'per_second') => setPricingModel(value)}
                    options={[
                      { value: 'per_token', label: 'Per Million Tokens' },
                      { value: 'per_second', label: 'Per Second' },
                    ]}
                  />
                </Form.Item>

                {pricingModel === 'per_token' ? (
                  <>
                    <Form.Item
                      label="Input Cost (per 1M tokens)"
                      name="input_cost_per_token"
                      rules={[{ validator: validateNumber }]}
                      className="mb-4"
                    >
                      <TextInput />
                    </Form.Item>
                    <Form.Item
                      label="Output Cost (per 1M tokens)"
                      name="output_cost_per_token"
                      rules={[{ validator: validateNumber }]}
                      className="mb-4"
                    >
                      <TextInput />
                    </Form.Item>
                  </>
                ) : (
                  <Form.Item
                    label="Cost Per Second"
                    name="input_cost_per_second"
                    rules={[{ validator: validateNumber }]}
                    className="mb-4"
                  >
                    <TextInput />
                  </Form.Item>
                )}
              </div>
            )}
            </div>

            {/* Security & Access Control Section */}
            <div className="border border-gray-200 rounded-lg p-4 mb-6">
              <div className="mb-4">
                <h5 className="text-base font-semibold text-gray-900 mb-1">Security & Access Control</h5>
                <p className={descriptionTextStyle}>Configure security policies and access restrictions</p>
              </div>

              <Form.Item 
                label={
                  <span>
                    Guardrails{' '}
                    <Tooltip title={
                      <div>
                        Apply safety guardrails to this model to filter content or enforce policies.{" "}
                        <a 
                          href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-400 hover:text-blue-300"
                        >
                          Learn more
                        </a>
                      </div>
                    }>
                      <InfoCircleOutlined className="text-gray-400" />
                    </Tooltip>
                  </span>
                }
                name="guardrails" 
                className="mb-4"
                help="Select existing guardrails. Go to 'Guardrails' tab to create new guardrails."
              >
                <Select
                  mode="tags"
                  style={{ width: '100%' }}
                  placeholder="Select or enter guardrails"
                  options={guardrailsList.map(name => ({ value: name, label: name }))}
                />
              </Form.Item>

              <Form.Item
                label="Use in pass through routes"
                name="use_in_pass_through"
                valuePropName="checked"
                className="mb-4"
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

              <CacheControlSettings
                form={form}
                showCacheControl={showCacheControl}
                onCacheControlChange={handleCacheControlChange}
              />
            </div>

            {/* Advanced Configuration Section */}
            <div className="border border-gray-200 rounded-lg p-4 mb-6">
              <div className="mb-4">
                <h5 className="text-base font-semibold text-gray-900 mb-1">Advanced Configuration</h5>
                <p className={descriptionTextStyle}>Raw JSON configuration for advanced use cases</p>
              </div>

              <Form.Item
                label="LiteLLM Params"
                name="litellm_extra_params"
                tooltip="Optional litellm params used for making a litellm.completion() call."
                className="mb-4"
                rules={[{ validator: validateJSON }]}
              >
                <TextArea
                  rows={4}
                  placeholder='{
  "stream_timeout": 60,
  "drop_params": true,
  "metadata": {"custom": "value"}
}'
                />
              </Form.Item>
              <Row className="mb-4">
                <Col span={10}></Col>
                <Col span={10}>
                  <Text className={descriptionTextStyle}>
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
  "mode": "chat",
  "description": "Custom model configuration"
}'
                />
              </Form.Item>
            </div>
          </div>
        </AccordionBody>
      </Accordion>
    </>
  );
};

export default AdvancedSettings;