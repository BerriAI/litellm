import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Modal, Form, Input, Select } from "antd";
import { Button, Title, Text, Grid, Col } from "@tremor/react";
import { DataTable } from "../view_logs/table";
import { searchToolColumns } from "./search_tool_columns";
import {
  fetchSearchTools,
  updateSearchTool,
  deleteSearchTool,
  fetchAvailableSearchProviders,
} from "../networking";
import { SearchTool, AvailableSearchProvider } from "./types";
import { isAdminRole } from "@/utils/roles";
import NotificationsManager from "../molecules/notifications_manager";
import { SearchToolView } from "./search_tool_view";
import CreateSearchTool from "./create_search_tool";

interface SearchToolsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const DeleteModal: React.FC<{
  isModalOpen: boolean;
  title: string;
  confirmDelete: () => void;
  cancelDelete: () => void;
}> = ({ isModalOpen, title, confirmDelete, cancelDelete }) => {
  if (!isModalOpen) return null;
  return (
    <Modal open={isModalOpen} onOk={confirmDelete} okType="danger" onCancel={cancelDelete}>
      <Grid numItems={1} className="gap-2 w-full">
        <Title>{title}</Title>
        <Col numColSpan={1}>
          <p>Are you sure you want to delete this search tool?</p>
        </Col>
      </Grid>
    </Modal>
  );
};

const SearchTools: React.FC<SearchToolsProps> = ({ accessToken, userRole, userID }) => {
  const {
    data: searchTools,
    isLoading: isLoadingTools,
    refetch,
  } = useQuery({
    queryKey: ["searchTools"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchSearchTools(accessToken).then((res) => res.search_tools || []);
    },
    enabled: !!accessToken,
  }) as { data: SearchTool[]; isLoading: boolean; refetch: () => void };

  const {
    data: providersResponse,
    isLoading: isLoadingProviders,
  } = useQuery({
    queryKey: ["searchProviders"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchAvailableSearchProviders(accessToken);
    },
    enabled: !!accessToken,
  }) as { data: { providers: AvailableSearchProvider[] }; isLoading: boolean };

  const availableProviders = providersResponse?.providers || [];

  // State
  const [toolIdToDelete, setToolToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [editTool, setEditTool] = useState(false);
  const [isCreateModalVisible, setCreateModalVisible] = useState(false);
  const [isEditModalVisible, setEditModalVisible] = useState(false);
  const [form] = Form.useForm();

  const columns = React.useMemo(
    () =>
      searchToolColumns(
        (toolId: string) => {
          setSelectedToolId(toolId);
          setEditTool(false);
        },
        (toolId: string) => {
          const tool = searchTools?.find((t) => t.search_tool_id === toolId);
          if (tool) {
            form.setFieldsValue({
              search_tool_name: tool.search_tool_name,
              search_provider: tool.litellm_params.search_provider,
              api_key: tool.litellm_params.api_key,
              api_base: tool.litellm_params.api_base,
              timeout: tool.litellm_params.timeout,
              max_retries: tool.litellm_params.max_retries,
              description: tool.search_tool_info?.description,
            });
            setSelectedToolId(toolId);
            setEditModalVisible(true);
          }
        },
        handleDelete,
        availableProviders,
      ),
    [availableProviders, searchTools],
  );

  function handleDelete(toolId: string) {
    setToolToDelete(toolId);
    setIsDeleteModalOpen(true);
  }

  const confirmDelete = async () => {
    if (toolIdToDelete == null || accessToken == null) {
      return;
    }
    try {
      await deleteSearchTool(accessToken, toolIdToDelete);
      NotificationsManager.success("Deleted search tool successfully");
      refetch();
    } catch (error) {
      console.error("Error deleting the search tool:", error);
      NotificationsManager.error("Failed to delete search tool");
    }
    setIsDeleteModalOpen(false);
    setToolToDelete(null);
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setToolToDelete(null);
  };

  const handleCreateSuccess = (newSearchTool: SearchTool) => {
    setCreateModalVisible(false);
    refetch();
  };

  const handleEditSubmit = async () => {
    if (!accessToken || !selectedToolId) return;

    try {
      const values = await form.validateFields();
      const searchToolData = {
        search_tool_name: values.search_tool_name,
        litellm_params: {
          search_provider: values.search_provider,
          api_key: values.api_key,
          api_base: values.api_base,
          timeout: values.timeout ? parseFloat(values.timeout) : undefined,
          max_retries: values.max_retries ? parseInt(values.max_retries) : undefined,
        },
        search_tool_info: values.description ? {
          description: values.description,
        } : undefined,
      };

      await updateSearchTool(accessToken, selectedToolId, searchToolData);
      NotificationsManager.success("Search tool updated successfully");
      setEditModalVisible(false);
      form.resetFields();
      setSelectedToolId(null);
      refetch();
    } catch (error) {
      console.error("Failed to update search tool:", error);
      NotificationsManager.error("Failed to update search tool");
    }
  };

  const renderEditForm = () => (
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
        <Select placeholder="Select a search provider" loading={isLoadingProviders}>
          {availableProviders.map((provider) => (
            <Select.Option key={provider.provider_name} value={provider.provider_name}>
              {provider.ui_friendly_name}
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      <Form.Item name="api_key" label="API Key" extra="API key for the search provider">
        <Input.Password placeholder="Enter API key" />
      </Form.Item>

      <Form.Item name="description" label="Description">
        <Input.TextArea rows={3} placeholder="Description of this search tool" />
      </Form.Item>
    </Form>
  );

  if (!accessToken || !userRole || !userID) {
    console.log("Missing required authentication parameters", { accessToken, userRole, userID });
    return <div className="p-6 text-center text-gray-500">Missing required authentication parameters.</div>;
  }

  const ToolsTab = () =>
    selectedToolId ? (
      <SearchToolView
        searchTool={
          searchTools?.find((tool: SearchTool) => tool.search_tool_id === selectedToolId) || {
            search_tool_id: "",
            search_tool_name: "",
            litellm_params: {
              search_provider: "",
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
        <div className="w-full px-6 mt-6">
          <DataTable
            data={searchTools || []}
            columns={columns}
            renderSubComponent={() => <div></div>}
            getRowCanExpand={() => false}
            isLoading={isLoadingTools}
            noDataMessage="No search tools configured"
          />
        </div>
      </div>
    );

  return (
    <div className="w-full h-full p-6">
      <DeleteModal
        isModalOpen={isDeleteModalOpen}
        title="Delete Search Tool"
        confirmDelete={confirmDelete}
        cancelDelete={cancelDelete}
      />

      <CreateSearchTool
        userRole={userRole}
        accessToken={accessToken}
        onCreateSuccess={handleCreateSuccess}
        isModalVisible={isCreateModalVisible}
        setModalVisible={setCreateModalVisible}
      />

      {/* Edit Modal */}
      <Modal
        title="Edit Search Tool"
        open={isEditModalVisible}
        onOk={handleEditSubmit}
        onCancel={() => {
          setEditModalVisible(false);
          form.resetFields();
          setSelectedToolId(null);
        }}
        width={600}
      >
        {renderEditForm()}
      </Modal>

      <Title>Search Tools</Title>
      <Text className="text-tremor-content mt-2">Configure and manage your search providers</Text>
      {isAdminRole(userRole) && (
        <Button className="mt-4 mb-4" onClick={() => setCreateModalVisible(true)}>
          + Add New Search Tool
        </Button>
      )}

      <ToolsTab />
    </div>
  );
};

export default SearchTools;
