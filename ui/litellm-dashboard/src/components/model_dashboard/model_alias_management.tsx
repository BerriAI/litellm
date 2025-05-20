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
  Typography,
} from "antd";
import {
  PencilIcon,
  TrashIcon,
  PlusIcon,
  InformationCircleIcon,
  RefreshIcon,
} from "@heroicons/react/outline";
import { getRouterSettings, updateRouterSettings, modelAvailableCall } from "../networking";

const { Text: AntText } = Typography;

interface ModelAliasManagementProps {
  accessToken: string;
  onRefresh: () => void;
  userID?: string;
  userRole?: string;
}

const ModelAliasManagement: React.FC<ModelAliasManagementProps> = ({
  accessToken,
  onRefresh,
  userID,
  userRole,
}) => {
  const [modelGroupAliases, setModelGroupAliases] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [formLoading, setFormLoading] = useState<boolean>(false);
  const [addForm] = Form.useForm();
  const [editForm] = Form.useForm();
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  
  // Modal state
  const [isEditModalVisible, setIsEditModalVisible] = useState<boolean>(false);
  const [editingAlias, setEditingAlias] = useState<string | null>(null);

  // Fetch router settings
  useEffect(() => {
    fetchRouterSettings();
  }, [accessToken]);

  // Fetch available models
  useEffect(() => {
    fetchAvailableModels();
  }, [accessToken, userID, userRole]);

  const fetchAvailableModels = async () => {
    if (!accessToken || !userID || !userRole) return;

    try {
      const model_available = await modelAvailableCall(
        accessToken,
        userID,
        userRole
      );
      
      const available_model_names = model_available["data"].map(
        (element: { id: string }) => element.id
      );
      
      console.log("available_model_names for aliases:", available_model_names);
      setAvailableModels(available_model_names);
    } catch (error) {
      console.error("Error fetching available models:", error);
      message.error("Failed to load available models");
    }
  };

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
    <div className="bg-white rounded-lg">
      <Title>Model Group Aliases</Title>
      <Text className="text-tremor-content text-sm">
        Map model alias names to actual model names. Aliases can be used interchangeably with the original model names in API calls.
      </Text>
      <Card className="shadow-md rounded-lg mt-6 border border-gray-200">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-base font-medium text-gray-700 m-0">Configured Aliases</h3>
          <Tooltip title="Refresh aliases">
            <Button 
              type="default" 
              icon={<RefreshIcon className="h-4 w-4 text-blue-600" />} 
              onClick={fetchRouterSettings} 
              disabled={isLoading}
              className="border-blue-200 hover:border-blue-400 hover:bg-blue-50 transition-colors"
            />
          </Tooltip>
        </div>
        
        <Divider className="my-5" />
        
        {/* Add New Alias Form */}
        <Form
          form={addForm}
          layout="horizontal"
          onFinish={handleAddModelGroupAlias}
          className="mb-8 p-6 bg-gray-50 rounded-md border border-gray-200 shadow-sm"
          initialValues={{ aliasName: "", targetModel: "" }}
        >
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            <div className="md:col-span-5">
              <Form.Item
                name="aliasName"
                label={<span className="font-medium text-gray-700">Alias Name</span>}
                rules={[{ 
                  required: true,
                  message: "Please enter an alias name"
                }]}
              >
                <Input 
                  placeholder="Enter alias name" 
                  disabled={formLoading}
                  className="rounded-md border-gray-300 hover:border-blue-400 focus:border-blue-500 transition-colors"
                  suffix={
                    <Tooltip title="This name will be used in API calls as an alternative to the actual model name">
                      <InformationCircleIcon className="h-4 w-4 text-gray-400" />
                    </Tooltip>
                  }
                />
              </Form.Item>
            </div>
            
            <div className="md:col-span-5">
              <Form.Item
                name="targetModel"
                label={<span className="font-medium text-gray-700">Target Model</span>}
                rules={[{ 
                  required: true,
                  message: "Please select a target model"
                }]}
              >
                <Select
                  placeholder="Select model"
                  disabled={formLoading}
                  showSearch
                  optionFilterProp="children"
                  className="rounded-md border-gray-300"
                  dropdownStyle={{ borderRadius: '0.375rem' }}
                >
                  {availableModels.map((model, idx) => (
                    <Select.Option key={idx} value={model}>
                      {model}
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
            </div>
            
            <div className="md:col-span-2 flex items-end">
              <Form.Item className="w-full">
                <TremorButton 
                  variant="primary"
                  loading={formLoading}
                  className="w-full justify-center shadow-sm"
                >
                  Add Alias
                </TremorButton>
              </Form.Item>
            </div>
          </div>
        </Form>
        
        <Spin spinning={isLoading} tip="Loading aliases...">
          <div className="border border-gray-200 rounded-md overflow-hidden shadow-sm">
            <Table className="mt-0">
              <TableHead className="bg-gray-50 border-b border-gray-200">
                <TableRow>
                  <TableHeaderCell className="py-4 font-semibold text-gray-700">Alias Name</TableHeaderCell>
                  <TableHeaderCell className="py-4 font-semibold text-gray-700">Target Model</TableHeaderCell>
                  <TableHeaderCell className="py-4 font-semibold text-gray-700 w-1/6 text-right pr-6">Actions</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(modelGroupAliases).map(([alias, model], idx) => (
                  <TableRow key={idx} className="hover:bg-gray-50 transition-colors border-b border-gray-100 last:border-none">
                    <TableCell className="py-4 font-medium text-gray-800">{alias}</TableCell>
                    <TableCell className="py-4 text-gray-600 font-mono text-sm">{model}</TableCell>
                    <TableCell className="py-4 text-right">
                      <Space>
                        <Tooltip title="Edit alias">
                          <Button
                            type="text"
                            icon={<PencilIcon className="h-4 w-4 text-blue-600" />}
                            onClick={() => openEditModal(alias)}
                            disabled={formLoading}
                            className="hover:bg-blue-50 rounded-md"
                          />
                        </Tooltip>
                        <Tooltip title="Delete alias">
                          <Button
                            type="text"
                            icon={<TrashIcon className="h-4 w-4 text-red-600" />}
                            onClick={() => handleDeleteModelGroupAlias(alias)}
                            disabled={formLoading}
                            className="hover:bg-red-50 rounded-md"
                          />
                        </Tooltip>
                      </Space>
                    </TableCell>
                  </TableRow>
                ))}
                {Object.keys(modelGroupAliases).length === 0 && (
                  <TableRow>
                    <TableCell colSpan={3} className="py-16">
                      <Empty 
                        description={<span className="text-gray-500">No model aliases defined</span>}
                        image={Empty.PRESENTED_IMAGE_SIMPLE}
                        className="my-6"
                      >
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
          title={<span className="text-lg font-semibold">Edit Model Alias</span>}
          open={isEditModalVisible}
          onCancel={closeEditModal}
          footer={null}
          maskClosable={!formLoading}
          closable={!formLoading}
          className="rounded-lg"
          width={520}
        >
          <Form
            form={editForm}
            layout="vertical"
            onFinish={handleUpdateModelGroupAlias}
            initialValues={{ aliasName: "", targetModel: "" }}
            className="mt-4"
          >
            <Form.Item
              name="aliasName"
              label={<span className="font-medium text-gray-700">Alias Name</span>}
              rules={[{ 
                required: true,
                message: "Please enter an alias name"
              }]}
              className="mb-4"
            >
              <Input 
                placeholder="Enter alias name" 
                disabled={formLoading}
                className="rounded-md border-gray-300"
                suffix={
                  <Tooltip title="This name will be used in API calls as an alternative to the actual model name">
                    <InformationCircleIcon className="h-4 w-4 text-gray-400" />
                  </Tooltip>
                }
              />
            </Form.Item>
            
            <Form.Item
              name="targetModel"
              label={<span className="font-medium text-gray-700">Target Model</span>}
              rules={[{ 
                required: true,
                message: "Please select a target model"
              }]}
              className="mb-6"
            >
              <Select
                placeholder="Select model"
                disabled={formLoading}
                showSearch
                optionFilterProp="children"
                className="rounded-md"
              >
                {availableModels.map((model, idx) => (
                  <Select.Option key={idx} value={model}>
                    {model}
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>
            
            <div className="flex justify-end gap-3">
              <TremorButton
                variant="secondary"
                onClick={closeEditModal}
                disabled={formLoading}
                className="shadow-sm hover:bg-gray-100"
              >
                Cancel
              </TremorButton>
              <TremorButton
                variant="primary"
                loading={formLoading}
                className="shadow-sm"
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