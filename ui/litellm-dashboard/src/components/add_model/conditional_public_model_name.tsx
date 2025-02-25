import React, { useEffect } from "react";
import { Form, Table, Input } from "antd";
import { Text, TextInput } from "@tremor/react";
import { Row, Col } from "antd";

const ConditionalPublicModelName: React.FC = () => {
  // Access the form instance
  const form = Form.useFormInstance();

  // Watch the 'model' field for changes
  const selectedModels = Form.useWatch('model', form) || [];
  const showPublicModelName = !selectedModels.includes('all-wildcard');

  // Auto-populate model mappings when selected models change
  useEffect(() => {
    if (selectedModels.length > 0 && !selectedModels.includes('all-wildcard')) {
      const mappings = selectedModels.map((model: string) => ({
        public_name: model,
        litellm_model: model
      }));
      form.setFieldValue('model_mappings', mappings);
    }
  }, [selectedModels, form]);

  if (!showPublicModelName) return null;

  const columns = [
    {
      title: 'Public Name',
      dataIndex: 'public_name',
      key: 'public_name',
      render: (text: string, record: any, index: number) => {
        return (
          <TextInput
            defaultValue={text}
            onChange={(e) => {
              const newMappings = [...form.getFieldValue('model_mappings')];
              newMappings[index].public_name = e.target.value;
              form.setFieldValue('model_mappings', newMappings);
            }}
          />
        );
      }
    },
    {
      title: 'LiteLLM Model',
      dataIndex: 'litellm_model',
      key: 'litellm_model',
    }
  ];

  return (
    <>
      <Form.Item
        label="Model Mappings"
        name="model_mappings"
        tooltip="Map public model names to LiteLLM model names for load balancing"
        labelCol={{ span: 10 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        required={true}
      >
        <Table 
          dataSource={form.getFieldValue('model_mappings')} 
          columns={columns} 
          pagination={false}
          size="small"
        />
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