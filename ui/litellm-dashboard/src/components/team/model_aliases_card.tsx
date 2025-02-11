import React, { useState } from "react";
import {
  Card,
  Title,
  Text,
  Button as TremorButton,
} from "@tremor/react";
import { Modal, Form, Select, Input, message } from "antd";
import { teamUpdateCall } from "@/components/networking";

interface ModelAliasesCardProps {
  teamId: string;
  accessToken: string | null;
  currentAliases: Record<string, string>;
  availableModels: string[];
  onUpdate: () => void;
}

const ModelAliasesCard: React.FC<ModelAliasesCardProps> = ({
  teamId,
  accessToken,
  currentAliases,
  availableModels,
  onUpdate,
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();

  const handleCreateAlias = async (values: any) => {
    try {
      if (!accessToken) return;

      const newAliases = {
        ...currentAliases,
        [values.original_model]: values.alias_name,
      };

      const updateData = {
        team_id: teamId,
        model_aliases: newAliases,
      };

      await teamUpdateCall(accessToken, updateData);
      message.success("Model alias created successfully");
      setIsModalVisible(false);
      form.resetFields();
      onUpdate();
    } catch (error) {
      message.error("Failed to create model alias");
      console.error("Error creating model alias:", error);
    }
  };

  return (
    <div className="mt-8">
      <Text className="text-gray-600 mb-4">
        Below, please find current model aliases, and guidance on when they will be updated to new versions (if guidance is available).
      </Text>
      
      <div className="bg-white rounded-lg p-6 border border-gray-200">
        <div className="flex justify-between items-center mb-6">
          <div>
            <div className="flex space-x-4 text-gray-600">
              <div className="w-64">ALIAS</div>
              <div>POINTS TO</div>
            </div>
          </div>
          <TremorButton
            size="md"
            variant="secondary"
            onClick={() => setIsModalVisible(true)}
            className="bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          >
            Create Model Alias
          </TremorButton>
        </div>

        <div className="space-y-4">
          {Object.entries(currentAliases).map(([originalModel, aliasName], index) => (
            <div key={index} className="flex space-x-4 border-t border-gray-200 pt-4">
              <div className="w-64">
                <span className="bg-gray-100 px-2 py-1 rounded font-mono text-sm text-gray-700">
                  {aliasName}
                </span>
              </div>
              <div>
                <span className="bg-gray-100 px-2 py-1 rounded font-mono text-sm text-gray-700">
                  {originalModel}
                </span>
              </div>
            </div>
          ))}
          {Object.keys(currentAliases).length === 0 && (
            <div className="text-gray-500 text-center py-4 border-t border-gray-200">
              No model aliases configured
            </div>
          )}
        </div>
      </div>

      <Modal
        title="Create Model Alias"
        open={isModalVisible}
        onCancel={() => {
          setIsModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={500}
      >
        <Form
          form={form}
          onFinish={handleCreateAlias}
          layout="vertical"
          className="mt-4"
        >
          <Form.Item
            label="Alias Name"
            name="alias_name"
            rules={[{ required: true, message: "Please enter an alias name" }]}
          >
            <Input 
              placeholder="Enter the model alias (e.g., gpt-4o)"
              className="w-full font-mono"
            />
          </Form.Item>

          <Form.Item
            label="Points To"
            name="original_model"
            rules={[{ required: true, message: "Please select a model" }]}
          >
            <Select 
              placeholder="Select model version"
              className="w-full font-mono"
              showSearch
              optionFilterProp="children"
            >
              {availableModels.map((model) => (
                <Select.Option key={model} value={model} className="font-mono">
                  {model}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>

          <div className="flex justify-end gap-2 mt-6">
            <TremorButton
              size="md"
              variant="secondary"
              onClick={() => {
                setIsModalVisible(false);
                form.resetFields();
              }}
              className="bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
            >
              Cancel
            </TremorButton>
            <TremorButton
              size="md"
              variant="primary"
              type="submit"
              className="bg-blue-600 text-white hover:bg-blue-700"
            >
              Create Alias
            </TremorButton>
          </div>
        </Form>
      </Modal>
    </div>
  );
};

export default ModelAliasesCard;