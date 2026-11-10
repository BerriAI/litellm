import { Page, PageContent } from "@/components/shared/Page";
import { AccessGroupResponse, useAccessGroups } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { useDeleteAccessGroup } from "@/app/(dashboard)/hooks/accessGroups/useDeleteAccessGroup";
import { Boxes, Plus, SearchIcon, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { AccessGroupDetail } from "./AccessGroupsDetailsPage";
import { AccessGroupCreateDialog } from "./access-group-create/AccessGroupCreateDialog";
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
    skillNames: r.access_skill_names ?? [],
    keyIds: r.assigned_key_ids,
    teamIds: r.assigned_team_ids,
    createdAt: r.created_at,
    createdBy: r.created_by ?? "",
    updatedAt: r.updated_at,
    updatedBy: r.updated_by ?? "",
  };
}

export function AccessGroupsPage() {
  const { t } = useTranslation();
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
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <Boxes />
          {t("accessGroups.page.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("accessGroups.page.description")}</PageHeaderDescription>
        {canModify && (
          <PageHeaderControls>
            <Button onClick={() => setIsCreateModalVisible(true)}>
              <Plus className="size-4" />
              {t("accessGroups.page.create")}
            </Button>
          </PageHeaderControls>
        )}
      </PageHeader>

      <PageContent className="gap-3">
        <div className="flex items-center">
          <InputGroup className="max-w-[400px]">
            <InputGroupAddon>
              <SearchIcon className="size-4 text-muted-foreground" />
            </InputGroupAddon>
            <InputGroupInput
              placeholder={t("accessGroups.page.searchPlaceholder")}
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
            />
            {searchText && (
              <InputGroupAddon align="inline-end">
                <InputGroupButton
                  size="icon-xs"
                  aria-label={t("accessGroups.page.clearSearch")}
                  onClick={() => setSearchText("")}
                >
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
      </PageContent>

      <AccessGroupCreateDialog open={isCreateModalVisible} onOpenChange={setIsCreateModalVisible} />

      <DeleteResourceModal
        isOpen={!!groupToDelete}
        title={t("accessGroups.page.deleteModal.title")}
        message={t("accessGroups.page.deleteModal.message")}
        resourceInformationTitle={t("accessGroups.page.deleteModal.informationTitle")}
        resourceInformation={[
          { label: t("accessGroups.page.deleteModal.id"), value: groupToDelete?.id, code: true },
          { label: t("accessGroups.page.deleteModal.name"), value: groupToDelete?.name },
          { label: t("accessGroups.page.deleteModal.description"), value: groupToDelete?.description || "—" },
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
    </Page>
  );
}
