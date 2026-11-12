import type { SkillListQuery } from "@/components/claude_code_plugins/types";
import { SkillFilters } from "@/components/claude_code_plugins/SkillFilters";
import { useSkillFilters, skillSortQuery } from "@/components/claude_code_plugins/skillCatalog";
import React, { useState } from "react";
import { useQuery, useQueryClient, type QueryKey } from "@tanstack/react-query";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { listSkillHubSkills, getSkillHubSkill, deleteSkillHubSkill } from "@/components/networking";
import AddPluginForm from "./add_plugin_form";
import PluginTable from "./PluginTable";
import SkillDetail, { skillHubDetail } from "@/components/claude_code_plugins/skill_detail";
import { isProxyAdminRole, isProxyAdminTierRole, isViewOnlySessionRole } from "@/utils/roles";
import { toast } from "@/lib/toast";
import { isPendingSkill, type SkillHubSkill, type SkillMutationResponse } from "@/components/claude_code_plugins/types";
import { useUrlTab } from "@/hooks/useUrlTab";
import SkillSubmissions from "./SkillSubmissions";

interface ClaudeCodePluginsPanelProps {
  accessToken: string | null;
  userRole?: string;
  isViewOnly?: boolean;
}

const scanResultFilter = (scan: string): boolean | undefined => (scan === "all" ? undefined : scan === "yes");

const SKILLS_TABS = ["skills", "submissions"] as const;

function skillPermissions(accessToken: string | null, userRole: string, isViewOnly: boolean) {
  const readOnly =
    isViewOnly || isViewOnlySessionRole(userRole) || ["Admin Viewer", "Internal Viewer"].includes(userRole);
  const canWrite = Boolean(accessToken && userRole) && !readOnly;
  return {
    readOnly,
    canWrite,
    isAdmin: isProxyAdminRole(userRole) && canWrite,
    adminView: isProxyAdminTierRole(userRole),
  };
}

interface CatalogSkillDetailProps {
  skill: SkillHubSkill;
  accessToken: string | null;
  queryScope: QueryKey;
  canWrite: boolean;
  isAdmin: boolean;
  onBack: () => void;
  onEdit: (skill: SkillHubSkill) => void;
}

function CatalogSkillDetail({
  skill,
  accessToken,
  queryScope,
  canWrite,
  isAdmin,
  onBack,
  onEdit,
}: CatalogSkillDetailProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const detailKey = [...queryScope, "detail", skill.id, skill.name];
  const detail = useQuery({
    queryKey: detailKey,
    queryFn: () => getSkillHubSkill(accessToken!, skill.name),
    enabled: Boolean(accessToken),
  });
  return (
    <>
      {detail.isError && (
        <div role="alert" className="text-sm text-destructive">
          {t("skills.panel.detailFailed")}
          <Button variant="link" onClick={() => void detail.refetch()}>
            {t("skills.submissions.retry")}
          </Button>
        </div>
      )}
      <SkillDetail
        skill={skillHubDetail(detail.data ?? skill)}
        onBack={onBack}
        accessToken={accessToken}
        isAdmin={isAdmin}
        onRestored={(restored) => queryClient.setQueryData(detailKey, restored)}
        onEditClick={canWrite && detail.isSuccess && !detail.isFetching ? () => onEdit(detail.data) : undefined}
      />
    </>
  );
}

const SkillsPanelContent: React.FC<ClaudeCodePluginsPanelProps> = ({
  accessToken,
  userRole = "",
  isViewOnly = false,
}) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [tab, setTab] = useUrlTab(SKILLS_TABS, "skills");
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [editingSkill, setEditingSkill] = useState<SkillHubSkill>();
  const [isDeleting, setIsDeleting] = useState(false);
  const [pluginToDelete, setPluginToDelete] = useState<{
    name: string;
    displayName: string;
  } | null>(null);
  const [selectedSkillId, setSelectedSkillId] = useState<string | null>(null);
  const { readOnly, canWrite, isAdmin, adminView } = skillPermissions(accessToken, userRole, isViewOnly);
  const queryScope = ["skill-hub", accessToken, userRole, readOnly] as const;
  const showForm = Boolean(isAddModalVisible || editingSkill) && canWrite;
  const filters = useSkillFilters();
  const { pagination, onPaginationChange, sorting, onSortingChange } = filters;
  const { pageIndex, pageSize } = pagination;
  const catalogQuery: SkillListQuery & { limit: number; offset: number } = {
    limit: pageSize,
    offset: pageIndex * pageSize,
    query: filters.search,
    tag: filters.tag,
    security_scan_passed: scanResultFilter(filters.scan),
    ...skillSortQuery(sorting),
  };
  const plugins = useQuery({
    queryKey: [...queryScope, "published", catalogQuery],
    queryFn: () => listSkillHubSkills(accessToken!, catalogQuery),
    enabled: Boolean(accessToken),
  });
  const pluginsList = plugins.data?.skills ?? [];
  const selectedSkill = pluginsList.find((plugin) => plugin.id === selectedSkillId);
  const refreshSkills = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryScope, refetchType: "all" }),
      queryClient.invalidateQueries({ queryKey: ["skill-versions"], refetchType: "all" }),
    ]);
  };
  const handleSaved = async (response: SkillMutationResponse) => {
    if (isPendingSkill(response)) {
      await queryClient.invalidateQueries({ queryKey: [...queryScope, "submissions"], refetchType: "all" });
      return;
    }
    await refreshSkills();
  };

  const handleDeleteClick = (pluginName: string, displayName: string) => {
    setPluginToDelete({ name: pluginName, displayName });
  };

  const handleDeleteConfirm = async () => {
    if (!pluginToDelete || !accessToken || !isAdmin) return;

    setIsDeleting(true);
    try {
      await deleteSkillHubSkill(accessToken, pluginToDelete.name);
      toast.success(t("skills.panel.toasts.deleted", { name: pluginToDelete.displayName }));
      await refreshSkills();
    } catch (error) {
      console.error("Error deleting skill:", error);
      toast.error(t("skills.panel.toasts.deleteFailed"));
    } finally {
      setIsDeleting(false);
      setPluginToDelete(null);
    }
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      {selectedSkill ? (
        <CatalogSkillDetail
          skill={selectedSkill}
          isAdmin={isAdmin}
          accessToken={accessToken}
          queryScope={queryScope}
          canWrite={canWrite}
          onBack={() => setSelectedSkillId(null)}
          onEdit={setEditingSkill}
        />
      ) : (
        <>
          <div className="flex flex-col gap-2 mb-4">
            <h1 className="text-2xl font-bold">{t("skills.panel.title")}</h1>
            <p className="text-sm text-muted-foreground">{t("skills.panel.intro")}</p>
            <div className="mt-2 flex gap-2">
              <Button onClick={() => setIsAddModalVisible(true)} disabled={!canWrite}>
                {t("skills.panel.addButton")}
              </Button>
            </div>
          </div>

          <Tabs value={tab} onValueChange={(value) => setTab(value === "submissions" ? "submissions" : "skills")}>
            <TabsList variant="line">
              <TabsTrigger value="skills">{t("skills.panel.tabs.skills")}</TabsTrigger>
              <TabsTrigger value="submissions">
                {t(adminView ? "skills.panel.tabs.submissions" : "skills.panel.tabs.mySubmissions")}
              </TabsTrigger>
            </TabsList>
            <TabsContent value="skills">
              <div className="my-4">
                <SkillFilters {...filters} tags={plugins.data?.tags ?? []} />
              </div>
              {plugins.isError && (
                <div role="alert" className="my-3 text-sm text-destructive">
                  {t("skills.panel.loadFailed")}
                  <Button variant="link" onClick={() => void plugins.refetch()}>
                    {t("skills.submissions.retry")}
                  </Button>
                </div>
              )}
              <PluginTable
                pluginsList={pluginsList}
                isLoading={plugins.isLoading}
                filtered={filters.active}
                sorting={sorting}
                onSortingChange={onSortingChange}
                onDeleteClick={handleDeleteClick}
                isAdmin={isAdmin}
                onPluginClick={setSelectedSkillId}
              />
              <div className="mt-4 flex items-center justify-end gap-3">
                <span>{t("skills.submissions.page", { page: pageIndex + 1 })}</span>
                <Button
                  variant="outline"
                  disabled={pageIndex === 0 || plugins.isFetching}
                  onClick={() => onPaginationChange({ pageIndex: pageIndex - 1, pageSize })}
                >
                  {t("skills.submissions.previous")}
                </Button>
                <Button
                  variant="outline"
                  disabled={plugins.data?.next_offset == null || plugins.isFetching || plugins.isError}
                  onClick={() => onPaginationChange({ pageIndex: pageIndex + 1, pageSize })}
                >
                  {t("skills.submissions.next")}
                </Button>
              </div>
            </TabsContent>
            <TabsContent value="submissions">
              {accessToken && (
                <SkillSubmissions
                  accessToken={accessToken}
                  queryScope={queryScope}
                  canReview={isAdmin}
                  canWrite={canWrite}
                  adminView={adminView}
                  onChanged={refreshSkills}
                />
              )}
            </TabsContent>
          </Tabs>
        </>
      )}

      {showForm && (
        <AddPluginForm
          key={editingSkill ? editingSkill.id : "new"}
          visible
          skill={editingSkill}
          requiresApproval={!isAdmin}
          onClose={() => {
            setIsAddModalVisible(false);
            setEditingSkill(undefined);
          }}
          accessToken={accessToken}
          onSuccess={(response) => void handleSaved(response)}
        />
      )}

      {pluginToDelete && (
        <AlertDialog
          open
          onOpenChange={(open) => {
            if (!open) setPluginToDelete(null);
          }}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t("skills.panel.deleteDialog.title")}</AlertDialogTitle>
              <AlertDialogDescription>
                <Trans
                  i18nKey="skills.panel.deleteDialog.confirm"
                  components={{ name: <strong>{pluginToDelete.displayName}</strong> }}
                />
              </AlertDialogDescription>
              <p className="text-sm text-muted-foreground">{t("skills.panel.deleteDialog.irreversible")}</p>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("skills.panel.deleteDialog.cancel")}</AlertDialogCancel>
              <Button variant="destructive" onClick={handleDeleteConfirm} disabled={isDeleting}>
                {t("skills.panel.deleteDialog.delete")}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};

const ClaudeCodePluginsPanel: React.FC<ClaudeCodePluginsPanelProps> = (props) => (
  <SkillsPanelContent key={`${props.accessToken}:${props.userRole}:${props.isViewOnly}`} {...props} />
);

export default ClaudeCodePluginsPanel;
