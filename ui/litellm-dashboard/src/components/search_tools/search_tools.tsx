import React, { useState, useEffect } from "react";
import { Card, Button, Table, Modal, Form, Input, Select, Typography, Space, Popconfirm, message } from "antd";
import { PlusOutlined, EditOutlined, DeleteOutlined, EyeOutlined } from "@ant-design/icons";
import {
  fetchSearchTools,
  createSearchTool,
  updateSearchTool,
  deleteSearchTool,
  fetchAvailableSearchProviders,
} from "../networking";
import { SearchTool, AvailableSearchProvider } from "./types";

const { Title, Text } = Typography;
const { TextArea } = Input;

interface SearchToolsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const SearchTools: React.FC<SearchToolsProps> = ({ accessToken }) => {
  const [searchTools, setSearchTools] = useState<SearchTool[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [viewModalVisible, setViewModalVisible] = useState(false);
  const [selectedTool, setSelectedTool] = useState<SearchTool | null>(null);
  const [availableProviders, setAvailableProviders] = useState<AvailableSearchProvider[]>([]);
  const [form] = Form.useForm();

  useEffect(() => {
    loadSearchTools();
    loadAvailableProviders();
  }, []);

  const loadSearchTools = async () => {
    if (!accessToken) return;
    
    setLoading(true);
    try {
      const response = await fetchSearchTools(accessToken);
      setSearchTools(response.search_tools || []);
    } catch (error) {
      message.error("Failed to load search tools");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const loadAvailableProviders = async () => {
    if (!accessToken) return;
    
    try {
      const response = await fetchAvailableSearchProviders(accessToken);
      setAvailableProviders(response.providers || []);
    } catch (error) {
      console.error("Failed to load providers:", error);
    }
  };

  const handleCreate = () => {
    form.resetFields();
    setCreateModalVisible(true);
  };

  const handleEdit = (tool: SearchTool) => {
    setSelectedTool(tool);
    form.setFieldsValue({
      search_tool_name: tool.search_tool_name,
      search_provider: tool.litellm_params.search_provider,
      api_key: tool.litellm_params.api_key,
      api_base: tool.litellm_params.api_base,
      timeout: tool.litellm_params.timeout,
      description: tool.search_tool_info?.description,
    });
    setEditModalVisible(true);
  };

  const handleView = (tool: SearchTool) => {
    setSelectedTool(tool);
    setViewModalVisible(true);
  };

  const handleDelete = async (searchToolId: string) => {
    if (!accessToken) return;
    
    try {
      await deleteSearchTool(accessToken, searchToolId);
      message.success("Search tool deleted successfully");
      loadSearchTools();
    } catch (error) {
      message.error("Failed to delete search tool");
      console.error(error);
    }
  };

  const handleCreateSubmit = async () => {
    if (!accessToken) return;
    
    try {
      const values = await form.validateFields();
      const searchToolData = {
        search_tool_name: values.search_tool_name,
        litellm_params: {
          search_provider: values.search_provider,
          api_key: values.api_key,
          api_base: values.api_base,
          timeout: values.timeout,
        },
        search_tool_info: {
          description: values.description,
        },
      };

      await createSearchTool(accessToken, searchToolData);
      message.success("Search tool created successfully");
      setCreateModalVisible(false);
      form.resetFields();
      loadSearchTools();
    } catch (error) {
      message.error("Failed to create search tool");
      console.error(error);
    }
  };

  const handleEditSubmit = async () => {
    if (!accessToken || !selectedTool) return;
    
    try {
      const values = await form.validateFields();
      const searchToolData = {
        search_tool_name: values.search_tool_name,
        litellm_params: {
          search_provider: values.search_provider,
          api_key: values.api_key,
          api_base: values.api_base,
          timeout: values.timeout,
        },
        search_tool_info: {
          description: values.description,
        },
      };

      await updateSearchTool(accessToken, selectedTool.search_tool_id!, searchToolData);
      message.success("Search tool updated successfully");
      setEditModalVisible(false);
      form.resetFields();
      setSelectedTool(null);
      loadSearchTools();
    } catch (error) {
      message.error("Failed to update search tool");
      console.error(error);
    }
  };

  const columns = [
    {
      title: "Name",
      dataIndex: "search_tool_name",
      key: "search_tool_name",
      render: (text: string, record: SearchTool) => (
        <Button type="link" onClick={() => handleView(record)}>
          {text}
        </Button>
      ),
    },
    {
      title: "Provider",
      dataIndex: ["litellm_params", "search_provider"],
      key: "search_provider",
      render: (provider: string) => {
        const providerInfo = availableProviders.find(p => p.provider_name === provider);
        return providerInfo?.ui_friendly_name || provider;
      },
    },
    {
      title: "API Base",
      dataIndex: ["litellm_params", "api_base"],
      key: "api_base",
      render: (text: string) => text || "-",
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      render: (text: string) => text ? new Date(text).toLocaleString() : "-",
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: SearchTool) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />} 
            onClick={() => handleView(record)}
          >
            View
          </Button>
          <Button 
            size="small" 
            icon={<EditOutlined />} 
            onClick={() => handleEdit(record)}
          >
            Edit
          </Button>
          <Popconfirm
            title="Are you sure you want to delete this search tool?"
            onConfirm={() => handleDelete(record.search_tool_id!)}
            okText="Yes"
            cancelText="No"
          >
            <Button size="small" danger icon={<DeleteOutlined />}>
              Delete
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const renderForm = () => (
    <Form form={form} layout="vertical">
      <Form.Item
        name="search_tool_name"
        label="Search Tool Name"
        rules={[{ required: true, message: "Please enter a search tool name" }]}
      >
        <Input placeholder="e.g., my-perplexity-search" />
      </Form.Item>

      <Form.Item
        name="search_provider"
        label="Search Provider"
        rules={[{ required: true, message: "Please select a search provider" }]}
      >
        <Select placeholder="Select a search provider">
          {availableProviders.map((provider) => (
            <Select.Option key={provider.provider_name} value={provider.provider_name}>
              {provider.ui_friendly_name}
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      <Form.Item
        name="api_key"
        label="API Key"
        extra="API key for the search provider"
      >
        <Input.Password placeholder="Enter API key" />
      </Form.Item>

      <Form.Item
        name="api_base"
        label="API Base URL (Optional)"
      >
        <Input placeholder="Custom API base URL" />
      </Form.Item>

      <Form.Item
        name="timeout"
        label="Timeout (seconds)"
      >
        <Input type="number" placeholder="30" />
      </Form.Item>

      <Form.Item
        name="description"
        label="Description"
      >
        <TextArea rows={3} placeholder="Description of this search tool" />
      </Form.Item>
    </Form>
  );

  return (
    <div style={{ padding: "24px" }}>
      <Card>
        <div style={{ marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <Title level={3} style={{ margin: 0 }}>Search Tools</Title>
            <Text type="secondary">
              Manage search providers for web search capabilities
            </Text>
          </div>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleCreate}
          >
            Add Search Tool
          </Button>
        </div>

        <Table
          dataSource={searchTools}
          columns={columns}
          rowKey="search_tool_id"
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* Create Modal */}
      <Modal
        title="Create Search Tool"
        open={createModalVisible}
        onOk={handleCreateSubmit}
        onCancel={() => {
          setCreateModalVisible(false);
          form.resetFields();
        }}
        width={600}
      >
        {renderForm()}
      </Modal>

      {/* Edit Modal */}
      <Modal
        title="Edit Search Tool"
        open={editModalVisible}
        onOk={handleEditSubmit}
        onCancel={() => {
          setEditModalVisible(false);
          form.resetFields();
          setSelectedTool(null);
        }}
        width={600}
      >
        {renderForm()}
      </Modal>

      {/* View Modal */}
      <Modal
        title="Search Tool Details"
        open={viewModalVisible}
        onCancel={() => {
          setViewModalVisible(false);
          setSelectedTool(null);
        }}
        footer={[
          <Button key="close" onClick={() => setViewModalVisible(false)}>
            Close
          </Button>,
          <Button 
            key="edit" 
            type="primary" 
            onClick={() => {
              setViewModalVisible(false);
              handleEdit(selectedTool!);
            }}
          >
            Edit
          </Button>,
        ]}
        width={700}
      >
        {selectedTool && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>Name:</Text> {selectedTool.search_tool_name}
            </div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>Provider:</Text> {selectedTool.litellm_params.search_provider}
            </div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>API Key:</Text> {selectedTool.litellm_params.api_key ? "****" : "Not set"}
            </div>
            {selectedTool.litellm_params.api_base && (
              <div style={{ marginBottom: 16 }}>
                <Text strong>API Base:</Text> {selectedTool.litellm_params.api_base}
              </div>
            )}
            {selectedTool.litellm_params.timeout && (
              <div style={{ marginBottom: 16 }}>
                <Text strong>Timeout:</Text> {selectedTool.litellm_params.timeout}s
              </div>
            )}
            {selectedTool.search_tool_info?.description && (
              <div style={{ marginBottom: 16 }}>
                <Text strong>Description:</Text>
                <div style={{ marginTop: 8 }}>{selectedTool.search_tool_info.description}</div>
              </div>
            )}
            <div style={{ marginBottom: 16 }}>
              <Text strong>Created:</Text> {selectedTool.created_at ? new Date(selectedTool.created_at).toLocaleString() : "-"}
            </div>
            <div>
              <Text strong>Updated:</Text> {selectedTool.updated_at ? new Date(selectedTool.updated_at).toLocaleString() : "-"}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default SearchTools;

