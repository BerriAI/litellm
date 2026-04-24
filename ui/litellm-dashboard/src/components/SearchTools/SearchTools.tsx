import { isAdminRole } from "@/utils/roles";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Loader2 } from "lucide-react";
import React, { useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import NotificationsManager from "../molecules/notifications_manager";
import {
  deleteSearchTool,
  fetchAvailableSearchProviders,
  fetchSearchTools,
  updateSearchTool,
} from "../networking";
import { DataTable } from "../view_logs/table";
import CreateSearchTool from "./CreateSearchTools";
import { searchToolColumns } from "./SearchToolColumn";
import { SearchToolView } from "./SearchToolView";
import { AvailableSearchProvider, SearchTool } from "./types";

interface SearchToolsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

interface EditFormValues {
  search_tool_name: string;
  search_provider: string;
  api_key?: string;
  api_base?: string;
  timeout?: string;
  max_retries?: string;
  description?: string;
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

  const { data: providersResponse, isLoading: isLoadingProviders } = useQuery({
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
  const [isDeleting, setIsDeleting] = useState(false);
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [editTool, setEditTool] = useState(false);
  const [isCreateModalVisible, setCreateModalVisible] = useState(false);
  const [isEditModalVisible, setEditModalVisible] = useState(false);

  const editForm = useForm<EditFormValues>({
    defaultValues: {
      search_tool_name: "",
      search_provider: "",
      api_key: "",
      api_base: "",
      timeout: "",
      max_retries: "",
      description: "",
    },
  });

  function handleDelete(toolId: string) {
    setToolToDelete(toolId);
    setIsDeleteModalOpen(true);
  }

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
            editForm.reset({
              search_tool_name: tool.search_tool_name,
              search_provider: tool.litellm_params.search_provider,
              api_key: tool.litellm_params.api_key ?? "",
              api_base: tool.litellm_params.api_base ?? "",
              timeout: tool.litellm_params.timeout !== undefined ? String(tool.litellm_params.timeout) : "",
              max_retries:
                tool.litellm_params.max_retries !== undefined ? String(tool.litellm_params.max_retries) : "",
              description: tool.search_tool_info?.description ?? "",
            });
            setSelectedToolId(toolId);
            setEditModalVisible(true);
          }
        },
        handleDelete,
        availableProviders,
      ),
    [availableProviders, searchTools, editForm],
  );

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

  const handleCreateSuccess = (_newSearchTool: SearchTool) => {
    setCreateModalVisible(false);
    refetch();
  };

  const handleEditCancel = () => {
    setEditModalVisible(false);
    editForm.reset();
    setSelectedToolId(null);
  };

  const handleEditSubmit = editForm.handleSubmit(async (values) => {
    if (!accessToken || !selectedToolId) return;

    try {
      const searchToolData = {
        search_tool_name: values.search_tool_name,
        litellm_params: {
          search_provider: values.search_provider,
          api_key: values.api_key,
          api_base: values.api_base,
          timeout: values.timeout ? parseFloat(values.timeout) : undefined,
          max_retries: values.max_retries ? parseInt(values.max_retries) : undefined,
        },
        search_tool_info: values.description
          ? {
              description: values.description,
            }
          : undefined,
      };

      await updateSearchTool(accessToken, selectedToolId, searchToolData);
      NotificationsManager.success("Search tool updated successfully");
      handleEditCancel();
      refetch();
    } catch (error) {
      console.error("Failed to update search tool:", error);
      NotificationsManager.error("Failed to update search tool");
    }
  });

  if (!accessToken || !userRole || !userID) {
    return (
      <div className="p-6 text-center text-muted-foreground">Missing required authentication parameters.</div>
    );
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
      <div className="w-full h-full relative">
        {isLoadingTools && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50 z-10">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}
        <DataTable
          columns={columns}
          data={searchTools || []}
          renderSubComponent={() => <></>}
          getRowCanExpand={() => false}
          isLoading={false}
          noDataMessage="No search tools configured"
        />
      </div>
    );

  return (
    <div className="w-full h-full p-6">
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

      {/* Edit Modal */}
      <Dialog open={isEditModalVisible} onOpenChange={(o) => (!o ? handleEditCancel() : undefined)}>
        <DialogContent className="max-w-[600px]">
          <DialogHeader>
            <DialogTitle>Edit Search Tool</DialogTitle>
          </DialogHeader>
          <FormProvider {...editForm}>
            <form onSubmit={handleEditSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="edit-search-tool-name">
                  Search Tool Name <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="edit-search-tool-name"
                  placeholder="e.g., my-perplexity-search"
                  {...editForm.register("search_tool_name", {
                    required: "Please enter a search tool name",
                  })}
                />
                {editForm.formState.errors.search_tool_name && (
                  <p className="text-sm text-destructive">
                    {editForm.formState.errors.search_tool_name.message as string}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="edit-search-provider">
                  Search Provider <span className="text-destructive">*</span>
                </Label>
                <Controller
                  control={editForm.control}
                  name="search_provider"
                  rules={{ required: "Please select a search provider" }}
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange} disabled={isLoadingProviders}>
                      <SelectTrigger id="edit-search-provider">
                        <SelectValue placeholder="Select a search provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {availableProviders.map((provider) => (
                          <SelectItem key={provider.provider_name} value={provider.provider_name}>
                            {provider.ui_friendly_name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
                {editForm.formState.errors.search_provider && (
                  <p className="text-sm text-destructive">
                    {editForm.formState.errors.search_provider.message as string}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="edit-api-key">API Key</Label>
                <Input
                  id="edit-api-key"
                  type="password"
                  placeholder="Enter API key"
                  {...editForm.register("api_key")}
                />
                <p className="text-xs text-muted-foreground">API key for the search provider</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="edit-description">Description</Label>
                <Textarea
                  id="edit-description"
                  rows={3}
                  placeholder="Description of this search tool"
                  {...editForm.register("description")}
                />
              </div>

              <DialogFooter>
                <Button type="button" variant="outline" onClick={handleEditCancel}>
                  Cancel
                </Button>
                <Button type="submit">OK</Button>
              </DialogFooter>
            </form>
          </FormProvider>
        </DialogContent>
      </Dialog>

      <h1 className="text-2xl font-semibold">Search Tools</h1>
      <p className="text-muted-foreground mt-2">Configure and manage your search providers</p>
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
