import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { organizationKeys, useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useQueryClient } from "@tanstack/react-query";
import { useVisitedTabs } from "@/hooks/useVisitedTabs";
import { MoneyCell } from "@/components/shared/table_cells";
import CopyButton from "@/components/shared/CopyButton";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { teamDetailHref } from "@/utils/entityLinks";
import { createTeamAliasMap } from "@/utils/teamUtils";
import { BadgeLink } from "@/components/shared/BadgeLink";
import type { ColumnsType } from "antd/es/table";
import { ArrowLeft } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import MemberTable from "../common_components/MemberTable";
import UserSearchModal from "../common_components/user_search_modal";
import NotificationsManager from "../molecules/notifications_manager";
import {
  Member,
  organizationMemberAddCall,
  organizationMemberDeleteCall,
  organizationMemberUpdateCall,
} from "../networking";
import ObjectPermissionsView from "../object_permissions_view";
import MemberModal from "../team/EditMembership";
import { OrgSettingsForm } from "./org-settings/OrgSettingsForm";

interface OrganizationInfoProps {
  organizationId: string;
  onClose: () => void;
  accessToken: string | null;
  is_org_admin: boolean;
  is_proxy_admin: boolean;
  userModels: string[];
  editOrg: boolean;
}

const OrganizationInfoView: React.FC<OrganizationInfoProps> = ({
  organizationId,
  onClose,
  accessToken,
  is_org_admin,
  is_proxy_admin,
  userModels,
  editOrg,
}) => {
  const { t, i18n } = useTranslation("gateway");
  const queryClient = useQueryClient();
  const { data: orgData, isLoading: loading } = useOrganization(organizationId);
  const [isEditing, setIsEditing] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(null);
  const canEditOrg = is_org_admin || is_proxy_admin;
  const { data: teams } = useTeams();
  const { onTabChange, hasVisited } = useVisitedTabs(editOrg ? "settings" : "overview");

  const teamAliasMap = useMemo(() => createTeamAliasMap(teams), [teams]);
  const locale = i18n.resolvedLanguage === "ru" ? "ru-RU" : "en-US";

  const handleMemberAdd = async (values: any) => {
    try {
      if (accessToken == null) {
        return;
      }

      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };
      await organizationMemberAddCall(accessToken, organizationId, member);

      NotificationsManager.success(t("organizations.notifications.memberAdded"));
      setIsAddMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend(t("organizations.notifications.memberAddFailed"));
      console.error("Error adding organization member:", error);
    }
  };

  const handleMemberUpdate = async (values: any) => {
    try {
      if (!accessToken) return;

      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };

      await organizationMemberUpdateCall(accessToken, organizationId, member);
      NotificationsManager.success(t("organizations.notifications.memberUpdated"));
      setIsEditMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend(t("organizations.notifications.memberUpdateFailed"));
      console.error("Error updating organization member:", error);
    }
  };

  const handleMemberDelete = async (values: any) => {
    try {
      if (!accessToken) return;

      await organizationMemberDeleteCall(accessToken, organizationId, values.user_id);
      NotificationsManager.success(t("organizations.notifications.memberDeleted"));
      setIsEditMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend(t("organizations.notifications.memberDeleteFailed"));
      console.error("Error deleting organization member:", error);
    }
  };

  if (loading) {
    return <div className="p-4">{t("organizations.details.loading")}</div>;
  }

  if (!orgData) {
    return <div className="p-4">{t("organizations.details.notFound")}</div>;
  }

  const orgExtraColumns: ColumnsType<Member> = [
    {
      title: t("organizations.table.spend"),
      key: "spend",
      render: (_: unknown, record: Member) => {
        const orgMember =
          record.user_id != null ? (orgData.members || []).find((m) => m.user_id === record.user_id) : undefined;
        return <MoneyCell value={orgMember?.spend} decimals={4} />;
      },
    },
    {
      title: t("organizations.details.createdAt"),
      key: "created_at",
      render: (_: unknown, record: Member) => {
        const orgMember =
          record.user_id != null ? (orgData.members || []).find((m) => m.user_id === record.user_id) : undefined;
        return <span>{orgMember?.created_at ? new Date(orgMember.created_at).toLocaleString(locale) : "-"}</span>;
      },
    },
  ];

  return (
    <div className="h-screen w-full bg-background p-4">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <Button variant="ghost" onClick={onClose} className="mb-4">
            <ArrowLeft className="size-4" />
            {t("organizations.details.back")}
          </Button>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">{orgData.organization_alias}</h1>
          <div className="flex items-center gap-1">
            <span className="font-mono text-sm text-muted-foreground">{orgData.organization_id}</span>
            <CopyButton
              value={orgData.organization_id}
              label={t("organizations.details.copyId")}
              iconClassName="size-3"
            />
          </div>
        </div>
      </div>

      <Tabs defaultValue={editOrg ? "settings" : "overview"} onValueChange={onTabChange} className="mb-4">
        <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            {t("organizations.details.overview")}
          </TabsTrigger>
          <TabsTrigger value="members" className="flex-none rounded-none px-4 py-2">
            {t("organizations.details.members")}
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex-none rounded-none px-4 py-2">
            {t("organizations.details.settings")}
          </TabsTrigger>
        </TabsList>

        <TabsContent keepMounted={hasVisited("overview")} value="overview" className="pt-4">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">{t("organizations.details.organizationDetails")}</p>
                <div className="mt-2 text-sm text-foreground">
                  <p>
                    {t("organizations.details.created")}: {new Date(orgData.created_at).toLocaleDateString(locale)}
                  </p>
                  <p>
                    {t("organizations.details.updated")}: {new Date(orgData.updated_at).toLocaleDateString(locale)}
                  </p>
                  <p>
                    {t("organizations.details.createdBy")}: {orgData.created_by}
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">{t("organizations.details.budgetStatus")}</p>
                <div className="mt-2 text-sm text-foreground">
                  <p className="text-xl font-semibold">${formatNumberWithCommas(orgData.spend, 4)}</p>
                  <p>
                    {t("organizations.details.of")}{" "}
                    {orgData.litellm_budget_table.max_budget === null
                      ? t("organizations.table.unlimited")
                      : `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`}
                  </p>
                  {orgData.litellm_budget_table.budget_duration && (
                    <p className="text-muted-foreground">
                      {t("organizations.details.reset")}: {orgData.litellm_budget_table.budget_duration}
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">{t("organizations.details.rateLimits")}</p>
                <div className="mt-2 text-sm text-foreground">
                  <p>TPM: {orgData.litellm_budget_table.tpm_limit || t("organizations.table.unlimited")}</p>
                  <p>RPM: {orgData.litellm_budget_table.rpm_limit || t("organizations.table.unlimited")}</p>
                  {orgData.litellm_budget_table.max_parallel_requests && (
                    <p>
                      {t("organizations.details.maxParallelRequests")}:{" "}
                      {orgData.litellm_budget_table.max_parallel_requests}
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">{t("organizations.table.models")}</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {orgData.models.length === 0 ? (
                    <BadgeLink>{t("organizations.details.allProxyModels")}</BadgeLink>
                  ) : (
                    orgData.models.map((model, index) => <BadgeLink key={index}>{model}</BadgeLink>)
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">{t("organizations.details.teams")}</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {orgData.teams?.map((team, index) => (
                    <BadgeLink key={index} href={teamDetailHref(team.team_id)}>
                      {teamAliasMap[team.team_id] || team.team_id}
                    </BadgeLink>
                  ))}
                </div>
              </CardContent>
            </Card>

            <ObjectPermissionsView
              objectPermission={orgData.object_permission}
              variant="card"
              accessToken={accessToken}
            />
          </div>
        </TabsContent>

        <TabsContent keepMounted={hasVisited("members")} value="members" className="pt-4">
          <div className="space-y-4">
            <MemberTable
              members={(orgData.members || []).map((m) => ({
                role: m.user_role || "",
                user_id: m.user_id,
                user_email: m.user_email,
              }))}
              canEdit={canEditOrg}
              onEdit={(member) => {
                setSelectedEditMember(member);
                setIsEditMemberModalVisible(true);
              }}
              onDelete={(member) => handleMemberDelete(member)}
              onAddMember={() => setIsAddMemberModalVisible(true)}
              roleColumnTitle={t("organizations.details.organizationRole")}
              extraColumns={orgExtraColumns}
              emptyText={t("organizations.details.noMembers")}
            />
          </div>
        </TabsContent>

        <TabsContent keepMounted={hasVisited("settings")} value="settings" className="pt-4">
          <Card className="max-h-[65vh] overflow-y-auto">
            <CardContent>
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-foreground">{t("organizations.details.settingsTitle")}</h2>
                {canEditOrg && !isEditing && (
                  <Button onClick={() => setIsEditing(true)}>{t("organizations.details.editSettings")}</Button>
                )}
              </div>

              {isEditing ? (
                <OrgSettingsForm
                  organizationId={organizationId}
                  org={orgData}
                  accessToken={accessToken || ""}
                  onCancel={() => setIsEditing(false)}
                  onSaved={() => setIsEditing(false)}
                />
              ) : (
                <div className="space-y-4 text-sm">
                  <div>
                    <p className="font-medium text-foreground">{t("organizations.form.organizationName")}</p>
                    <div>{orgData.organization_alias}</div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{t("organizations.table.organizationId")}</p>
                    <div className="font-mono">{orgData.organization_id}</div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{t("organizations.details.createdAt")}</p>
                    <div>{new Date(orgData.created_at).toLocaleString(locale)}</div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{t("organizations.table.models")}</p>
                    <div className="mt-1 flex flex-wrap gap-2">
                      {orgData.models.map((model, index) => (
                        <BadgeLink key={index}>{model}</BadgeLink>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{t("organizations.details.rateLimits")}</p>
                    <div>TPM: {orgData.litellm_budget_table.tpm_limit || t("organizations.table.unlimited")}</div>
                    <div>RPM: {orgData.litellm_budget_table.rpm_limit || t("organizations.table.unlimited")}</div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{t("budgets.title")}</p>
                    <div>
                      {t("organizations.details.max")}:{" "}
                      {orgData.litellm_budget_table.max_budget !== null
                        ? `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`
                        : t("organizations.details.noLimit")}
                    </div>
                    <div>
                      {t("organizations.details.reset")}:{" "}
                      {orgData.litellm_budget_table.budget_duration || t("organizations.details.never")}
                    </div>
                  </div>

                  <ObjectPermissionsView
                    objectPermission={orgData.object_permission}
                    variant="inline"
                    className="border-t pt-4"
                    accessToken={accessToken}
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <UserSearchModal
        isVisible={isAddMemberModalVisible}
        onCancel={() => setIsAddMemberModalVisible(false)}
        onSubmit={handleMemberAdd}
        accessToken={accessToken}
        title={t("organizations.details.addMember")}
        roles={[
          {
            label: "org_admin",
            value: "org_admin",
            description: t("organizations.details.orgAdminDescription"),
          },
          {
            label: "internal_user",
            value: "internal_user",
            description: t("organizations.details.internalUserDescription"),
          },
          {
            label: "internal_user_viewer",
            value: "internal_user_viewer",
            description: t("organizations.details.internalUserViewerDescription"),
          },
        ]}
        defaultRole="internal_user"
      />
      <MemberModal
        visible={isEditMemberModalVisible}
        onCancel={() => setIsEditMemberModalVisible(false)}
        onSubmit={handleMemberUpdate}
        initialData={selectedEditMember}
        mode="edit"
        config={{
          title: t("organizations.details.editMember"),
          showEmail: true,
          showUserId: true,
          roleOptions: [
            { label: t("organizations.details.roleOrgAdmin"), value: "org_admin" },
            { label: t("organizations.details.roleInternalUser"), value: "internal_user" },
            { label: t("organizations.details.roleInternalUserViewer"), value: "internal_user_viewer" },
          ],
        }}
      />
    </div>
  );
};

export default OrganizationInfoView;
