import { AccessGroupResponse, useAccessGroups } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { useDeleteAccessGroup } from "@/app/(dashboard)/hooks/accessGroups/useDeleteAccessGroup";
import { Plus, SearchIcon, X } from "lucide-react";
import { useMemo, useState } from "react";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { PageHeader } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { AccessGroupDetail } from "./AccessGroupsDetailsPage";
import { AccessGroupCreateModal } from "./AccessGroupsModal/AccessGroupCreateModal";
import { AccessGroupsTable } from "./AccessGroupsTable";
import { AccessGroup } from "./types";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";

function mapResponseToAccessGroup(r: AccessGroupResponse): AccessGroup {
  return {
    id: r.access_group_id,
    name: r.access_group_name,
    description: r.description ?? "",
    modelIds: r.access_model_names,
    mcpServerIds: r.access_mcp_server_ids,
    agentIds: r.access_agent_ids,
    keyIds: r.assigned_key_ids,
    teamIds: r.assigned_team_ids,
    createdAt: r.created_at,
    createdBy: r.created_by ?? "",
    updatedAt: r.updated_at,
    updatedBy: r.updated_by ?? "",
  };
}

export function AccessGroupsPage() {
  const { userRole } = useAuthorized();
  // Admin Viewer follows the read-parity rule: see access groups, no writes.
  const canModify = isProxyAdminRole(userRole ?? "");
  const { data: groupsData, isLoading } = useAccessGroups();
  const groups = useMemo(() => (groupsData ?? []).map(mapResponseToAccessGroup), [groupsData]);

  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [groupToDelete, setGroupToDelete] = useState<AccessGroup | null>(null);
  const deleteMutation = useDeleteAccessGroup();

  const filteredGroups = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    if (!query) return groups;
    return groups.filter(
      (group) =>
        group.name.toLowerCase().includes(query) ||
        group.id.toLowerCase().includes(query) ||
        group.description.toLowerCase().includes(query),
    );
  }, [groups, searchText]);

  if (selectedGroupId) {
    return <AccessGroupDetail accessGroupId={selectedGroupId} onBack={() => setSelectedGroupId(null)} />;
  }

  return (
    <div className="p-6 px-12">
      <div className="mb-4">
        <PageHeader
          title="Access Groups"
          subtitle="Manage resource permissions for your organization"
          actions={
            canModify ? (
              <Button onClick={() => setIsCreateModalVisible(true)}>
                <Plus className="size-4" />
                Create Access Group
              </Button>
            ) : undefined
          }
        />
      </div>

      <div className="mb-3 flex items-center">
        <InputGroup className="max-w-[400px]">
          <InputGroupAddon>
            <SearchIcon className="size-4 text-muted-foreground" />
          </InputGroupAddon>
          <InputGroupInput
            placeholder="Search groups by name, ID, or description..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
          {searchText && (
            <InputGroupAddon align="inline-end">
              <InputGroupButton size="icon-xs" aria-label="Clear search" onClick={() => setSearchText("")}>
                <X />
              </InputGroupButton>
            </InputGroupAddon>
          )}
        </InputGroup>
      </div>

      <AccessGroupsTable
        groups={filteredGroups}
        isLoading={isLoading}
        isFiltered={searchText.trim().length > 0}
        canModify={canModify}
        onGroupClick={setSelectedGroupId}
        onDeleteClick={setGroupToDelete}
      />

      <AccessGroupCreateModal visible={isCreateModalVisible} onCancel={() => setIsCreateModalVisible(false)} />

      <DeleteResourceModal
        isOpen={!!groupToDelete}
        title="Delete Access Group"
        message="Are you sure you want to delete this access group? This action cannot be undone."
        resourceInformationTitle="Access Group Information"
        resourceInformation={[
          { label: "ID", value: groupToDelete?.id, code: true },
          { label: "Name", value: groupToDelete?.name },
          { label: "Description", value: groupToDelete?.description || "—" },
        ]}
        onCancel={() => setGroupToDelete(null)}
        onOk={() => {
          if (!groupToDelete) return;
          deleteMutation.mutate(groupToDelete.id, {
            onSuccess: () => {
              setGroupToDelete(null);
            },
          });
        }}
        confirmLoading={deleteMutation.isPending}
      />
    </div>
  );
}
