import React, { useState, useEffect } from "react";
import { Form, Space, Table, Input, Tag } from "antd";
import { Text, TextInput } from "@tremor/react";

interface ModelMapping {
  litellmModel: string;
  publicName: string;
  key: string;
}

const ConditionalPublicModelName: React.FC = () => {
  const form = Form.useFormInstance();
  const selectedModels = Form.useWatch('model', form) || [];
  const [mappings, setMappings] = useState<ModelMapping[]>([]);

  // Sync mappings with selected models
  useEffect(() => {
    const validModels = selectedModels.filter(model => 
      model !== 'custom' && model !== 'all-wildcard'
    );

    // Remove mappings for models that are no longer selected
    const updatedMappings = mappings.filter(mapping =>
      validModels.includes(mapping.litellmModel)
    );

    // Add new mappings for newly selected models
    const existingModelSet = new Set(updatedMappings.map(m => m.litellmModel));
    const newMappings = validModels
      .filter(model => !existingModelSet.has(model))
      .map(model => ({
        key: Date.now().toString() + model,
        litellmModel: model,
        publicName: model
      }));

    // Update mappings state with both kept and new mappings
    setMappings([...updatedMappings, ...newMappings]);
  }, [selectedModels]);

  // Update form value whenever mappings change
  useEffect(() => {
    const modelNameValue = mappings
      .map(m => `${m.litellmModel}:${m.publicName}`)
      .join(',');
    form.setFieldsValue({ model_name: modelNameValue });
  }, [mappings]);

  const columns = [
    {
      title: 'Public Name',
      dataIndex: 'publicName',
      key: 'publicName',
      render: (text: string, record: ModelMapping) => (
        <TextInput 
          defaultValue={text}
          onChange={(e) => updateMapping(record.key, e.target.value)}
          style={{ width: '100%' }}
          placeholder="Enter public model name"
        />
      )
    },
    {
      title: 'LiteLLM Model',
      dataIndex: 'litellmModel',
      key: 'litellmModel',
      render: (text: string) => <Tag color="blue">{text}</Tag>
    }
  ];

  const updateMapping = (key: string, newPublicName: string) => {
    setMappings(mappings.map(m => 
      m.key === key ? { ...m, publicName: newPublicName } : m
    ));
  };

  // Don't show anything if no models are selected or if all-wildcard is selected
  if (selectedModels.includes('all-wildcard') || selectedModels.length === 0) {
    return null;
  }

  return (
    <Form.Item
      label="Model Mappings"
      required={false}
      className="mb-0"
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        <Table 
          columns={columns}
          dataSource={mappings}
          size="small"
          pagination={false}
          className="mt-2"
        />
        <Text className="mt-1" style={{ fontSize: '13px', color: '#666' }}>
          Map LiteLLM models to custom public names for load balancing and user requests
        </Text>
      </Space>
    </Form.Item>
  );
};

export default ConditionalPublicModelName;