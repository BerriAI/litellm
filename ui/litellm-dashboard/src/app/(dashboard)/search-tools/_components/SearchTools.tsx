import { isAdminRole } from "@/utils/roles";
import { useQuery } from "@tanstack/react-query";
import React, { useState } from "react";
import { z } from "zod/v4";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import {
  deleteSearchTool,
  fetchAvailableSearchProviders,
  fetchSearchTools,
  updateSearchTool,
} from "@/components/networking";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import CreateSearchTool from "./CreateSearchTools";
import { buildSearchToolPayload } from "./searchToolPayload";
import SearchToolTable from "./SearchToolTable";
import { SearchToolView } from "./SearchToolView";
import { AvailableSearchProvider, SearchTool } from "./types";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface SearchToolsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const editSearchToolShape = {
  search_tool_name: z.string().min(1, "Please enter a search tool name"),
  search_provider: z.string().min(1, "Please select a search provider"),
  api_key: z.string().nullish(),
  description: z.string().nullish(),
};

const editSearchToolSchema = z.object(editSearchToolShape);

type EditSearchToolFormValues = z.infer<typeof editSearchToolSchema>;

const EMPTY_EDIT_VALUES: EditSearchToolFormValues = { search_tool_name: "", search_provider: "" };

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

  const [toolIdToDelete, setToolToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [editTool, setEditTool] = useState(false);
  const [isCreateModalVisible, setCreateModalVisible] = useState(false);
  const [isEditModalVisible, setEditModalVisible] = useState(false);
  const form = useZodForm(editSearchToolSchema, { defaultValues: EMPTY_EDIT_VALUES });

  const handleView = (toolId: string) => {
    setSelectedToolId(toolId);
    setEditTool(false);
  };

  const handleEditOpen = (toolId: string) => {
    const tool = searchTools?.find((t) => t.search_tool_id === toolId);
    if (!tool) {
      return;
    }
    const editFormValues: EditSearchToolFormValues = {
      search_tool_name: tool.search_tool_name,
      search_provider: tool.litellm_params.search_provider,
      api_key: tool.litellm_params.api_key,
      description: tool.search_tool_info?.description,
    };
    form.reset(editFormValues);
    setSelectedToolId(toolId);
    setEditModalVisible(true);
  };

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
      toast.success("Deleted search tool successfully");
      setIsDeleteModalOpen(false);
      setToolToDelete(null);
      refetch();
    } catch (error) {
      console.error("Error deleting the search tool:", error);
      toast.error("Failed to delete search tool");
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

  const submitEdit = form.handleSubmit(
    async (values) => {
      if (!accessToken || !selectedToolId) return;

      try {
        await updateSearchTool(accessToken, selectedToolId, buildSearchToolPayload(values));
        toast.success("Search tool updated successfully");
        setEditModalVisible(false);
        form.reset(EMPTY_EDIT_VALUES);
        setSelectedToolId(null);
        refetch();
      } catch (error) {
        console.error("Failed to update search tool:", error);
        toast.error("Failed to update search tool");
      }
    },
    (errors) => {
      console.error("Failed to update search tool:", errors);
      toast.error("Failed to update search tool");
    },
  );

  const handleEditSubmit = () => {
    if (!accessToken || !selectedToolId) return;
    void submitEdit();
  };

  const renderEditForm = () => (
    <form onSubmit={(event) => event.preventDefault()}>
      <FieldGroup>
        <FormField control={form.control} name="search_tool_name" label="Search Tool Name">
          {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="e.g., my-perplexity-search" />}
        </FormField>

        <FormField control={form.control} name="search_provider" label="Search Provider">
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Select
              items={availableProviders.map((provider) => ({
                label: provider.ui_friendly_name,
                value: provider.provider_name,
              }))}
              value={value === "" ? null : value}
              onValueChange={(provider: string | null) => onChange(provider ?? "")}
            >
              <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy} className="w-full">
                <SelectValue placeholder="Select a search provider" />
                {isLoadingProviders && <UiLoadingSpinner className="size-4" />}
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
        </FormField>

        <FormField control={form.control} name="api_key" label="API Key" description="API key for the search provider">
          {({ ref, value, ...field }) => (
            <PasswordInput {...field} ref={ref} value={value ?? ""} placeholder="Enter API key" />
          )}
        </FormField>

        <FormField control={form.control} name="description" label="Description">
          {({ ref, value, ...field }) => (
            <Textarea {...field} ref={ref} value={value ?? ""} rows={3} placeholder="Description of this search tool" />
          )}
        </FormField>
      </FieldGroup>
    </form>
  );

  if (!accessToken || !userRole || !userID) {
    return <div className="p-6 text-center text-muted-foreground">Missing required authentication parameters.</div>;
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
        <SearchToolTable
          searchTools={searchTools || []}
          isLoading={isLoadingTools}
          availableProviders={availableProviders}
          onView={handleView}
          onEdit={handleEditOpen}
          onDelete={handleDelete}
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

      <Dialog
        open={isEditModalVisible}
        onOpenChange={(open) => {
          if (!open) {
            setEditModalVisible(false);
            form.reset(EMPTY_EDIT_VALUES);
            setSelectedToolId(null);
          }
        }}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>Edit Search Tool</DialogTitle>
          </DialogHeader>
          {renderEditForm()}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setEditModalVisible(false);
                form.reset(EMPTY_EDIT_VALUES);
                setSelectedToolId(null);
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleEditSubmit}>OK</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <h1 className="text-lg font-semibold text-foreground">Search Tools</h1>
      <p className="mt-2 text-sm text-muted-foreground">Configure and manage your search providers</p>
      {isAdminRole(userRole) && (
        <Button className="mt-4 mb-4" variant="outline" onClick={() => setCreateModalVisible(true)}>
          + Add New Search Tool
        </Button>
      )}

      <ToolsTab />
    </div>
  );
};

export default SearchTools;
