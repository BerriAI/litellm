import { isAdminRole } from "@/utils/roles";
import { LoadingOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Button, Text, Title } from "@tremor/react";
import { Form, Input, Modal, Select, Spin, Table } from "antd";
import React, { useState } from "react";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import NotificationsManager from "../molecules/notifications_manager";
import { deleteFetchTool, fetchAvailableFetchProviders, fetchFetchTools, updateFetchTool } from "../networking";
import CreateFetchTool from "./CreateFetchTool";
import { fetchToolColumns } from "./FetchToolColumn";
import { FetchToolView } from "./FetchToolView";
import { AvailableFetchProvider, FetchTool } from "./types";

interface FetchToolsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const FetchTools: React.FC<FetchToolsProps> = ({ accessToken, userRole, userID }) => {
  const {
    data: fetchTools,
    isLoading: isLoadingTools,
    refetch,
  } = useQuery({
    queryKey: ["fetchTools"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchFetchTools(accessToken).then((res) => res.fetch_tools || []);
    },
    enabled: !!accessToken,
  }) as { data: FetchTool[]; isLoading: boolean; refetch: () => void };

  const { data: providersResponse, isLoading: isLoadingProviders } = useQuery({
    queryKey: ["fetchProviders"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchAvailableFetchProviders(accessToken);
    },
    enabled: !!accessToken,
  }) as { data: { providers: AvailableFetchProvider[] }; isLoading: boolean };

  const availableProviders = providersResponse?.providers || [];

  // State
  const [toolIdToDelete, setToolToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [editTool, setEditTool] = useState(false);
  const [isCreateModalVisible, setCreateModalVisible] = useState(false);
  const [isEditModalVisible, setEditModalVisible] = useState(false);
  const [form] = Form.useForm();

  const columns = React.useMemo(
    () =>
      fetchToolColumns(
        (toolId: string) => {
          setSelectedToolId(toolId);
          setEditTool(false);
        },
        (toolId: string) => {
          const tool = fetchTools?.find((t) => t.fetch_tool_id === toolId);
          if (tool) {
            form.setFieldsValue({
              fetch_tool_name: tool.fetch_tool_name,
              fetch_provider: tool.litellm_params.fetch_provider,
              api_key: tool.litellm_params.api_key,
              api_base: tool.litellm_params.api_base,
              timeout: tool.litellm_params.timeout,
              max_retries: tool.litellm_params.max_retries,
              description: tool.fetch_tool_info?.description,
            });
            setSelectedToolId(toolId);
            setEditModalVisible(true);
          }
        },
        handleDelete,
        availableProviders,
      ),
    [availableProviders, fetchTools, form],
  );

  function handleDelete(toolId: string) {
    setToolToDelete(toolId);
    setIsDeleteModalOpen(true);
  }

  const confirmDelete = async () => {
    if (toolIdToDelete == null || accessToken == null) {
      return;
    }
    setIsDeleting(true);
    try {
      await deleteFetchTool(accessToken, toolIdToDelete);
      NotificationsManager.success("Deleted fetch tool successfully");
      setIsDeleteModalOpen(false);
      setToolToDelete(null);
      refetch();
    } catch (error) {
      console.error("Error deleting the fetch tool:", error);
      NotificationsManager.error("Failed to delete fetch tool");
    } finally {
      setIsDeleting(false);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setToolToDelete(null);
  };

  const toolToDeleteObj = fetchTools?.find((t) => t.fetch_tool_id === toolIdToDelete);
  const providerInfo = toolToDeleteObj
    ? availableProviders.find((p) => p.provider_name === toolToDeleteObj.litellm_params.fetch_provider)
    : null;

  const handleCreateSuccess = (newFetchTool: FetchTool) => {
    setCreateModalVisible(false);
    refetch();
  };

  const handleEditSubmit = async () => {
    if (!accessToken || !selectedToolId) return;

    try {
      const values = await form.validateFields();
      const fetchToolData = {
        fetch_tool_name: values.fetch_tool_name,
        litellm_params: {
          fetch_provider: values.fetch_provider,
          api_key: values.api_key,
          api_base: values.api_base,
          timeout: values.timeout ? parseFloat(values.timeout) : undefined,
          max_retries: values.max_retries ? parseInt(values.max_retries) : undefined,
        },
        fetch_tool_info: values.description
          ? {
              description: values.description,
            }
          : undefined,
      };

      await updateFetchTool(accessToken, selectedToolId, fetchToolData);
      NotificationsManager.success("Fetch tool updated successfully");
      setEditModalVisible(false);
      form.resetFields();
      setSelectedToolId(null);
      refetch();
    } catch (error) {
      console.error("Failed to update fetch tool:", error);
      NotificationsManager.error("Failed to update fetch tool");
    }
  };

  const renderEditForm = () => (
    <Form form={form} layout="vertical">
      <Form.Item
        name="fetch_tool_name"
        label="Fetch Tool Name"
        rules={[{ required: true, message: "Please enter a fetch tool name" }]}
      >
        <Input placeholder="e.g., my-firecrawl-fetch" />
      </Form.Item>

      <Form.Item
        name="fetch_provider"
        label="Fetch Provider"
        rules={[{ required: true, message: "Please select a fetch provider" }]}
      >
        <Select placeholder="Select a fetch provider" loading={isLoadingProviders}>
          {availableProviders.map((provider) => (
            <Select.Option key={provider.provider_name} value={provider.provider_name}>
              {provider.ui_friendly_name}
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      <Form.Item name="api_key" label="API Key" extra="API key for the fetch provider">
        <Input.Password placeholder="Enter API key" />
      </Form.Item>

      <Form.Item name="api_base" label="API Base URL" extra="Optional: Custom API base URL for self-hosted instances">
        <Input placeholder="https://api.firecrawl.dev" />
      </Form.Item>

      <Form.Item name="description" label="Description">
        <Input.TextArea rows={3} placeholder="Description of this fetch tool" />
      </Form.Item>
    </Form>
  );

  if (!accessToken || !userRole || !userID) {
    console.log("Missing required authentication parameters", { accessToken, userRole, userID });
    return <div className="p-6 text-center text-gray-500">Missing required authentication parameters.</div>;
  }

  const ToolsTab = () =>
    selectedToolId ? (
      <FetchToolView
        fetchTool={
          fetchTools?.find((tool: FetchTool) => tool.fetch_tool_id === selectedToolId) || {
            fetch_tool_id: "",
            fetch_tool_name: "",
            litellm_params: {
              fetch_provider: "",
            },
          }
        }
        onBack={() => {
          setEditTool(false);
          setSelectedToolId(null);
          refetch();
        }}
        isEditing={editTool}
        accessToken={accessToken}
        availableProviders={availableProviders}
      />
    ) : (
      <div className="w-full h-full">
        <Spin spinning={isLoadingTools} indicator={<LoadingOutlined spin />} size="large">
          <Table
            bordered
            dataSource={fetchTools || []}
            columns={columns}
            rowKey={(record) => record.fetch_tool_id || record.fetch_tool_name}
            pagination={false}
            locale={{
              emptyText: "No fetch tools configured",
            }}
            size="small"
          />
        </Spin>
      </div>
    );

  return (
    <div className="w-full h-full p-6">
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Fetch Tool"
        message="Are you sure you want to delete this fetch tool? This action cannot be undone."
        resourceInformationTitle="Fetch Tool Information"
        resourceInformation={[
          { label: "Fetch Tool Name", value: toolToDeleteObj?.fetch_tool_name || "" },
          {
            label: "Provider",
            value: providerInfo?.ui_friendly_name || toolToDeleteObj?.litellm_params?.fetch_provider || "",
          },
        ]}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
        isDeleting={isDeleting}
      />

      <div className="flex flex-col h-full" style={{ maxHeight: "100vh" }}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <Title className="text-lg">Fetch Tools</Title>
            <Text>Manage your fetch tools for web content extraction</Text>
          </div>
          <Button type="primary" onClick={() => setCreateModalVisible(true)} className="bg-blue-600 hover:bg-blue-700">
            + Create Fetch Tool
          </Button>
        </div>

        <div className="flex-1 overflow-hidden">
          <ToolsTab />
        </div>
      </div>

      {/* Create Modal */}
      <Modal
        title="Create Fetch Tool"
        open={isCreateModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
        width={800}
      >
        <CreateFetchTool
          accessToken={accessToken}
          availableProviders={availableProviders}
          onSuccess={handleCreateSuccess}
          onCancel={() => setCreateModalVisible(false)}
        />
      </Modal>

      {/* Edit Modal */}
      <Modal
        title="Edit Fetch Tool"
        open={isEditModalVisible}
        onCancel={() => {
          setEditModalVisible(false);
          form.resetFields();
          setSelectedToolId(null);
        }}
        onOk={handleEditSubmit}
        okText="Save Changes"
        width={800}
      >
        {renderEditForm()}
      </Modal>
    </div>
  );
};

export default FetchTools;
