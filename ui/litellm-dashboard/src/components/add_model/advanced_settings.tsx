import React from "react";
import { Form, Switch } from "antd";
import { Text, Button, Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
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