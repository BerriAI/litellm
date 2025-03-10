import React from "react";
import { Form, Switch, Select, Input } from "antd";
import { Text, Button, Accordion, AccordionHeader, AccordionBody, TextInput } from "@tremor/react";
import { Row, Col, Typography, Card } from "antd";
import TextArea from "antd/es/input/TextArea";
import { Team } from "../key_team_helpers/key_list";
import TeamDropdown from "../common_components/team_dropdown";
const { Link } = Typography;

interface AdvancedSettingsProps {
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams?: Team[] | null;
}

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({
  showAdvancedSettings,
  setShowAdvancedSettings,
  teams,
}) => {
  const [form] = Form.useForm();
  const [customPricing, setCustomPricing] = React.useState(false);
  const [pricingModel, setPricingModel] = React.useState<'per_token' | 'per_second'>('per_token');

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

  return (
    <>
      <Accordion className="mt-2 mb-4">
        <AccordionHeader>
          <b>Advanced Settings</b>
        </AccordionHeader>
        <AccordionBody>
          <div className="bg-white rounded-lg">
            <Form.Item
              label="Team"
              name="team_id"
              className="mb-4"
            >
              <TeamDropdown teams={teams} />
            </Form.Item>

            <Form.Item
              label="Custom Pricing"
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