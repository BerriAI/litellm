import React from "react";
import { Form } from "antd";
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
  return (
    <>

        <Accordion className="mt-2 mb-4">
          
          <AccordionHeader>
            <b>Advanced Settings</b>
          </AccordionHeader>
          <AccordionBody>
            <div className="bg-white rounded-lg">
              <Form.Item
                label="LiteLLM Params"
                name="litellm_extra_params"
                tooltip="Optional litellm params used for making a litellm.completion() call."
                className="mb-4"
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