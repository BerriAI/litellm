import React from "react";
import { Form } from "antd";
import { TextInput, Text } from "@tremor/react";
import { Row, Col } from "antd";

const ConditionalPublicModelName: React.FC = () => {
  // Access the form instance
  const form = Form.useFormInstance();

  // Watch the 'model' field for changes
  const selectedModels = Form.useWatch('model', form) || [];
  const showPublicModelName = !selectedModels.includes('all-wildcard');

  if (!showPublicModelName) return null;

  return (
    <>
      <Form.Item
        label="Public Model Name"
        name="model_name"
        tooltip="Model name your users will pass in. Also used for load-balancing, LiteLLM will load balance between all models with this public name."
        labelCol={{ span: 10 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        required={false}
        className="mb-0"
        rules={[
          ({ getFieldValue }) => ({
            validator(_, value) {
              const selectedModels = getFieldValue('model') || [];
              if (!selectedModels.includes('all-wildcard') || value) {
                return Promise.resolve();
              }
              return Promise.reject(new Error('Public Model Name is required unless "All Models" is selected.'));
            },
          }),
        ]}
      >
        <TextInput placeholder="my-gpt-4" />
      </Form.Item>
      <Row>
        <Col span={10}></Col>
        <Col span={10}>
          <Text className="mb-2">
            Model name your users will pass in.
          </Text>
        </Col>
      </Row>
    </>
  );
};

export default ConditionalPublicModelName;