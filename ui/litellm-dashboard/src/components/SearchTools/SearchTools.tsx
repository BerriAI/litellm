import { isAdminRole } from "@/utils/roles";
import { teamListCall, type TeamsResponse } from "@/app/(dashboard)/hooks/teams/useTeams";
import { PlusOutlined, SearchOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Button, Form, Input, Modal, Select, Table, Tabs } from "antd";
import React, { useMemo, useState } from "react";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import NotificationsManager from "../molecules/notifications_manager";
import {
  deleteSearchTool,
  fetchAvailableSearchProviders,
  fetchSearchTools,
  updateSearchTool,
} from "../networking";
import { AntDLoadingSpinner } from "../ui/AntDLoadingSpinner";
import CreateSearchTool from "./CreateSearchTools";
import { searchToolColumns } from "./SearchToolColumn";
import SearchToolTestPlayground from "./SearchToolTestPlayground";
import { SearchToolView } from "./SearchToolView";
import { AvailableSearchProvider, SearchTool } from "./types";

interface SearchToolsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

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

  // For non-admin users, fetch their teams to scope search tools
  const isAdmin = userRole ? isAdminRole(userRole) : false;
  const { data: userTeamsResponse } = useQuery({
    queryKey: ["userTeamsForSearchTools", userID],
    queryFn: () => {
      if (!accessToken || !userID) throw new Error("Missing auth");
      return teamListCall(accessToken, 1, 100, { userID }) as Promise<TeamsResponse>;
    },
    enabled: !!accessToken && !!userID && !isAdmin,
  });

  // Compute allowed search tool IDs from user's teams
  const scopedSearchTools = useMemo(() => {
    if (!searchTools) return [];
    if (isAdmin) return searchTools;
    if (!userTeamsResponse?.teams) return [];

    // Collect all search_tool IDs the user's teams grant access to
    const allowedIds = new Set<string>();
    let hasWildcard = false;
    for (const team of userTeamsResponse.teams) {
      const teamSearchTools = team.object_permission?.search_tools;
      if (!teamSearchTools) continue;
      if (teamSearchTools.includes("*")) {
        hasWildcard = true;
        break;
      }
      for (const id of teamSearchTools) {
        allowedIds.add(id);
      }
    }

    if (hasWildcard) return searchTools;
    if (allowedIds.size === 0) return [];
    return searchTools.filter(
      (tool) => allowedIds.has(tool.search_tool_id || tool.search_tool_name),
    );
  }, [searchTools, isAdmin, userTeamsResponse]);

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
        isAdmin,
      ),
    [availableProviders, searchTools, form, isAdmin],
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
      await deleteSearchTool(accessToken, toolIdToDelete);
      NotificationsManager.success("Deleted search tool successfully");
      setIsDeleteModalOpen(false);
      setToolToDelete(null);
      refetch();
    } catch (error) {
      console.error("Error deleting the search tool:", error);
      NotificationsManager.error("Failed to delete search tool");
    } finally {
      setIsDeleting(false);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setToolToDelete(null);
  };

  const toolToDelete = searchTools?.find((t) => t.search_tool_id === toolIdToDelete);
  const providerInfo = toolToDelete
    ? availableProviders.find((p) => p.provider_name === toolToDelete.litellm_params.search_provider)
    : null;

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
        <Select placeholder="Select a search provider" notFoundContent={isLoadingProviders ? "Loading providers..." : "No providers found"}>
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
    ) : isLoadingTools ? (
        <div className="flex justify-center items-center py-16">
          <AntDLoadingSpinner size="large" />
        </div>
    ) : (
        <Table
          bordered
          dataSource={scopedSearchTools}
          columns={columns}
          rowKey={(record) => record.search_tool_id || record.search_tool_name}
          pagination={false}
          locale={{
            emptyText: "No search tools configured",
          }}
          size="small"
        />
    );

  return (
    <div className="w-full mx-4 h-[75vh]">
      <div className="gap-2 p-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-xl font-semibold text-gray-900"><SearchOutlined style={{ marginRight: 8 }} />Search Tools</h1>
            <p className="text-sm text-gray-500 mt-1">Configure and manage your search providers</p>
          </div>
          {isAdminRole(userRole) && (
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
            >
              Add Search Tool
            </Button>
          )}
        </div>

        <Tabs
          defaultActiveKey="tools"
          items={[
            {
              key: "tools",
              label: "Search Tools",
              children: <ToolsTab />,
            },
            {
              key: "test",
              label: "Test Search Tools",
              children: (
                <SearchToolTestPlayground
                  searchTools={scopedSearchTools}
                  availableProviders={availableProviders}
                  isLoading={isLoadingTools}
                  accessToken={accessToken}
                />
              ),
            },
          ]}
        />
      </div>

      {/* Modals */}
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Search Tool"
        message="Are you sure you want to delete this search tool? This action cannot be undone."
        resourceInformationTitle="Search Tool Information"
        resourceInformation={
          toolToDelete
            ? [
              { label: "Name", value: toolToDelete.search_tool_name },
              { label: "ID", value: toolToDelete.search_tool_id, code: true },
              {
                label: "Provider",
                value: providerInfo?.ui_friendly_name || toolToDelete.litellm_params.search_provider,
              },
              { label: "Description", value: toolToDelete.search_tool_info?.description || "-" },
            ]
            : []
        }
        onCancel={cancelDelete}
        onOk={confirmDelete}
        confirmLoading={isDeleting}
      />

      <CreateSearchTool
        userRole={userRole}
        accessToken={accessToken}
        onCreateSuccess={handleCreateSuccess}
        isModalVisible={isCreateModalVisible}
        setModalVisible={setCreateModalVisible}
      />

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
    </div>
  );
};

export default SearchTools;
