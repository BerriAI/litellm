import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  TableBody,
  Button as TremorButton,
} from "@tremor/react";
import {
  Input,
  Select,
  message,
  Button,
  Form,
  Spin,
  Tooltip,
  Space,
  Empty,
  Divider,
  Modal,
} from "antd";
import {
  PencilIcon,
  TrashIcon,
  PlusIcon,
  InformationCircleIcon,
  RefreshIcon,
} from "@heroicons/react/outline";
import { getRouterSettings, updateRouterSettings } from "../networking";

interface ModelAliasManagementProps {
  accessToken: string;
  availableModels: string[];
  onRefresh: () => void;
}

const ModelAliasManagement: React.FC<ModelAliasManagementProps> = ({
  accessToken,
  availableModels,
  onRefresh,
}) => {
  const [modelGroupAliases, setModelGroupAliases] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [formLoading, setFormLoading] = useState<boolean>(false);
  const [addForm] = Form.useForm();
  const [editForm] = Form.useForm();
  
  // Modal state
  const [isEditModalVisible, setIsEditModalVisible] = useState<boolean>(false);
  const [editingAlias, setEditingAlias] = useState<string | null>(null);

  // Fetch router settings
  useEffect(() => {
    fetchRouterSettings();
  }, [accessToken]);

  const fetchRouterSettings = async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const settings = await getRouterSettings(accessToken);
      
      // Check both possible structures for model_group_alias
      if (settings?.router_settings?.model_group_alias) {
        setModelGroupAliases(settings.router_settings.model_group_alias);
      } else if (settings?.model_group_alias) {
        // Direct structure as shown in user's example
        setModelGroupAliases(settings.model_group_alias);
      } else {
        setModelGroupAliases({});
      }
    } catch (error) {
      console.error("Failed to fetch router settings:", error);
      message.error("Failed to load model aliases");
    } finally {
      setIsLoading(false);
    }
  };

  // Handle adding a new model group alias
  const handleAddModelGroupAlias = async (values: any) => {
    const { aliasName, targetModel } = values;
    
    if (!accessToken || !aliasName || !targetModel) {
      message.error("Alias name and model selection are required");
      return;
    }

    setFormLoading(true);
    try {
      const updatedAliases = { ...modelGroupAliases };
      updatedAliases[aliasName] = targetModel;
      
      await updateRouterSettings(accessToken, {
        model_group_alias: updatedAliases
      });
      
      setModelGroupAliases(updatedAliases);
      addForm.resetFields();
      
      message.success({
        content: "Model alias created successfully",
        key: 'aliasUpdate',
      });
      onRefresh();
    } catch (error) {
      console.error("Failed to save model group alias:", error);
      message.error({
        content: "Failed to create model alias",
        key: 'aliasError',
      });
    } finally {
      setFormLoading(false);
    }
  };

  // Handle updating a model group alias
  const handleUpdateModelGroupAlias = async (values: any) => {
    const { aliasName, targetModel } = values;
    
    if (!accessToken || !aliasName || !targetModel) {
      message.error("Alias name and model selection are required");
      return;
    }

    setFormLoading(true);
    try {
      const updatedAliases = { ...modelGroupAliases };
      
      // If alias name changed, remove the old one
      if (editingAlias && editingAlias !== aliasName) {
        delete updatedAliases[editingAlias];
      }
      
      updatedAliases[aliasName] = targetModel;
      
      await updateRouterSettings(accessToken, {
        model_group_alias: updatedAliases
      });
      
      setModelGroupAliases(updatedAliases);
      closeEditModal();
      
      message.success({
        content: "Model alias updated successfully",
        key: 'aliasUpdate',
      });
      onRefresh();
    } catch (error) {
      console.error("Failed to update model group alias:", error);
      message.error({
        content: "Failed to update model alias",
        key: 'aliasError',
      });
    } finally {
      setFormLoading(false);
    }
  };

  // Handle deleting a model group alias
  const handleDeleteModelGroupAlias = async (aliasName: string) => {
    setFormLoading(true);
    try {
      const updatedAliases = { ...modelGroupAliases };
      delete updatedAliases[aliasName];
      
      await updateRouterSettings(accessToken, {
        model_group_alias: updatedAliases
      });
      
      setModelGroupAliases(updatedAliases);
      message.success({
        content: "Model alias deleted successfully",
        key: 'aliasDelete',
      });
      onRefresh();
    } catch (error) {
      console.error("Failed to delete model group alias:", error);
      message.error({
        content: "Failed to delete model alias",
        key: 'aliasDeleteError',
      });
    } finally {
      setFormLoading(false);
    }
  };

  // Handle opening the edit modal
  const openEditModal = (aliasName: string) => {
    setEditingAlias(aliasName);
    editForm.setFieldsValue({
      aliasName: aliasName,
      targetModel: modelGroupAliases[aliasName] || "",
    });
    setIsEditModalVisible(true);
  };

  // Handle closing the edit modal
  const closeEditModal = () => {
    editForm.resetFields();
    setIsEditModalVisible(false);
    setEditingAlias(null);
  };

  return (
    <div>
    <Title>Model Group Aliases</Title>
    <Text className="text-tremor-content">
      Map model alias names to actual model names. Aliases can be used interchangeably with the original model names in API calls.
    </Text>
    <Card className="shadow-md rounded-lg mt-6">
      <div className="flex justify-between items-center mb-4">
        <Tooltip title="Reload aliases">
          <Button 
            type="default" 
            icon={<RefreshIcon className="h-4 w-4 text-blue-500" />} 
            onClick={fetchRouterSettings} 
            disabled={isLoading}
          />
        </Tooltip>
      </div>
      
      <Divider className="my-5" />
      
      {/* Add New Alias Form */}
      <Form
        form={addForm}
        layout="horizontal"
        onFinish={handleAddModelGroupAlias}
        className="mb-8 p-5 bg-gray-50 rounded-md border border-gray-200"
        initialValues={{ aliasName: "", targetModel: "" }}
      >
        <div className="flex flex-col md:flex-row gap-4">
          <Form.Item
            name="aliasName"
            label="Alias Name"
            className="mb-0 md:w-2/5"
            rules={[{ required: true, message: "Please enter an alias name" }]}
          >
            <Input 
              placeholder="Enter alias name" 
              disabled={formLoading}
              suffix={
                <Tooltip title="This name will be used in API calls as an alternative to the actual model name">
                  <InformationCircleIcon className="h-4 w-4 text-gray-400" />
                </Tooltip>
              }
            />
          </Form.Item>
          
          <Form.Item
            name="targetModel"
            label="Target Model"
            className="mb-0 md:w-2/5"
            rules={[{ required: true, message: "Please select a target model" }]}
          >
            <Select
              placeholder="Select model"
              disabled={formLoading}
              showSearch
              optionFilterProp="children"
            >
              {availableModels.map((model, idx) => (
                <Select.Option key={idx} value={model}>
                  {model}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          
          <div className="flex items-end gap-2 md:w-1/5">
            <Form.Item className="mb-0 flex-grow">
              <TremorButton 
                variant="primary"
                loading={formLoading}
              >
                Add Alias
              </TremorButton>
            </Form.Item>
          </div>
        </div>
      </Form>
      
      <Spin spinning={isLoading} tip="Loading aliases...">
        <div className="border rounded-md overflow-hidden">
          <Table className="mt-0">
            <TableHead className="bg-gray-50">
              <TableRow>
                <TableHeaderCell className="py-3 font-semibold text-gray-700">Alias Name</TableHeaderCell>
                <TableHeaderCell className="py-3 font-semibold text-gray-700">Target Model</TableHeaderCell>
                <TableHeaderCell className="py-3 font-semibold text-gray-700 w-1/6">Actions</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(modelGroupAliases).map(([alias, model], idx) => (
                <TableRow key={idx} className="hover:bg-gray-50 transition-colors">
                  <TableCell className="py-3 font-medium">{alias}</TableCell>
                  <TableCell className="py-3 text-gray-600">{model}</TableCell>
                  <TableCell className="py-3">
                    <Space>
                      <Tooltip title="Edit alias">
                        <Button
                          type="text"
                          icon={<PencilIcon className="h-4 w-4 text-blue-500" />}
                          onClick={() => openEditModal(alias)}
                          disabled={formLoading}
                          className="hover:bg-blue-50"
                        />
                      </Tooltip>
                      <Tooltip title="Delete alias">
                        <Button
                          type="text"
                          icon={<TrashIcon className="h-4 w-4 text-red-500" />}
                          onClick={() => handleDeleteModelGroupAlias(alias)}
                          disabled={formLoading}
                          className="hover:bg-red-50"
                        />
                      </Tooltip>
                    </Space>
                  </TableCell>
                </TableRow>
              ))}
              {Object.keys(modelGroupAliases).length === 0 && (
                <TableRow>
                  <TableCell colSpan={3} className="py-12">
                    <Empty 
                      description="No model aliases defined" 
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                    >
                      <Button 
                        type="primary" 
                        icon={<PlusIcon className="h-4 w-4" />}
                        onClick={() => addForm.setFieldsValue({ aliasName: "", targetModel: "" })}
                        disabled={formLoading}
                      >
                        Create New Alias
                      </Button>
                    </Empty>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </Spin>
      
      {/* Edit Modal */}
      <Modal
        title="Edit Model Alias"
        open={isEditModalVisible}
        onCancel={closeEditModal}
        footer={null}
        maskClosable={!formLoading}
        closable={!formLoading}
      >
        <Form
          form={editForm}
          layout="vertical"
          onFinish={handleUpdateModelGroupAlias}
          initialValues={{ aliasName: "", targetModel: "" }}
        >
          <Form.Item
            name="aliasName"
            label="Alias Name"
            rules={[{ required: true, message: "Please enter an alias name" }]}
          >
            <Input 
              placeholder="Enter alias name" 
              disabled={formLoading}
              suffix={
                <Tooltip title="This name will be used in API calls as an alternative to the actual model name">
                  <InformationCircleIcon className="h-4 w-4 text-gray-400" />
                </Tooltip>
              }
            />
          </Form.Item>
          
          <Form.Item
            name="targetModel"
            label="Target Model"
            rules={[{ required: true, message: "Please select a target model" }]}
          >
            <Select
              placeholder="Select model"
              disabled={formLoading}
              showSearch
              optionFilterProp="children"
            >
              {availableModels.map((model, idx) => (
                <Select.Option key={idx} value={model}>
                  {model}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          
          <div className="flex justify-end gap-2 mt-4">
            <TremorButton
              variant="secondary"
              onClick={closeEditModal}
              disabled={formLoading}
            >
              Cancel
            </TremorButton>
            <TremorButton
              variant="primary"
              loading={formLoading}
            >
              Update Alias
            </TremorButton>
          </div>
        </Form>
      </Modal>
    </Card>
    </div>
  );
};

export default ModelAliasManagement; 