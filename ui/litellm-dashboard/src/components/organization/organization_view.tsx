import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { organizationKeys, useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useQueryClient } from "@tanstack/react-query";
import { MoneyCell } from "@/components/shared/table_cells";
import CopyButton from "@/components/shared/CopyButton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { createTeamAliasMap } from "@/utils/teamUtils";
// MemberTable is shared with other routes and still consumes antd's column type.
import type { ColumnsType } from "antd/es/table";
import { ArrowLeft } from "lucide-react";
import React, { useMemo, useState } from "react";
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
  const queryClient = useQueryClient();
  const { data: orgData, isLoading: loading } = useOrganization(organizationId);
  const [isEditing, setIsEditing] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(null);
  const canEditOrg = is_org_admin || is_proxy_admin;
  const { data: teams } = useTeams();

  const teamAliasMap = useMemo(() => createTeamAliasMap(teams), [teams]);

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
      const response = await organizationMemberAddCall(accessToken, organizationId, member);

      NotificationsManager.success("Organization member added successfully");
      setIsAddMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend("Failed to add organization member");
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

      const response = await organizationMemberUpdateCall(accessToken, organizationId, member);
      NotificationsManager.success("Organization member updated successfully");
      setIsEditMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update organization member");
      console.error("Error updating organization member:", error);
    }
  };

  const handleMemberDelete = async (values: any) => {
    try {
      if (!accessToken) return;

      await organizationMemberDeleteCall(accessToken, organizationId, values.user_id);
      NotificationsManager.success("Organization member deleted successfully");
      setIsEditMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend("Failed to delete organization member");
      console.error("Error deleting organization member:", error);
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!orgData) {
    return <div className="p-4">Organization not found</div>;
  }

  const orgExtraColumns: ColumnsType<Member> = [
    {
      title: "Spend (USD)",
      key: "spend",
      render: (_: unknown, record: Member) => {
        const orgMember =
          record.user_id != null ? (orgData.members || []).find((m) => m.user_id === record.user_id) : undefined;
        return <MoneyCell value={orgMember?.spend} decimals={4} />;
      },
    },
    {
      title: "Created At",
      key: "created_at",
      render: (_: unknown, record: Member) => {
        const orgMember =
          record.user_id != null ? (orgData.members || []).find((m) => m.user_id === record.user_id) : undefined;
        return <span>{orgMember?.created_at ? new Date(orgMember.created_at).toLocaleString() : "-"}</span>;
      },
    },
  ];

  return (
    <div className="h-screen w-full bg-background p-4">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <Button variant="ghost" onClick={onClose} className="mb-4">
            <ArrowLeft className="size-4" />
            Back to Organizations
          </Button>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">{orgData.organization_alias}</h1>
          <div className="flex items-center gap-1">
            <span className="font-mono text-sm text-muted-foreground">{orgData.organization_id}</span>
            <CopyButton value={orgData.organization_id} label="Copy organization ID" iconClassName="size-3" />
          </div>
        </div>
      </div>

      <Tabs defaultValue={editOrg ? "settings" : "overview"} className="mb-4">
        <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            Overview
          </TabsTrigger>
          <TabsTrigger value="members" className="flex-none rounded-none px-4 py-2">
            Members
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex-none rounded-none px-4 py-2">
            Settings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="pt-4">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">Organization Details</p>
                <div className="mt-2 text-sm text-foreground">
                  <p>Created: {new Date(orgData.created_at).toLocaleDateString()}</p>
                  <p>Updated: {new Date(orgData.updated_at).toLocaleDateString()}</p>
                  <p>Created By: {orgData.created_by}</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">Budget Status</p>
                <div className="mt-2 text-sm text-foreground">
                  <p className="text-xl font-semibold">${formatNumberWithCommas(orgData.spend, 4)}</p>
                  <p>
                    of{" "}
                    {orgData.litellm_budget_table.max_budget === null
                      ? "Unlimited"
                      : `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`}
                  </p>
                  {orgData.litellm_budget_table.budget_duration && (
                    <p className="text-muted-foreground">Reset: {orgData.litellm_budget_table.budget_duration}</p>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">Rate Limits</p>
                <div className="mt-2 text-sm text-foreground">
                  <p>TPM: {orgData.litellm_budget_table.tpm_limit || "Unlimited"}</p>
                  <p>RPM: {orgData.litellm_budget_table.rpm_limit || "Unlimited"}</p>
                  {orgData.litellm_budget_table.max_parallel_requests && (
                    <p>Max Parallel Requests: {orgData.litellm_budget_table.max_parallel_requests}</p>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">Models</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {orgData.models.length === 0 ? (
                    <Badge variant="secondary">All proxy models</Badge>
                  ) : (
                    orgData.models.map((model, index) => (
                      <Badge key={index} variant="secondary">
                        {model}
                      </Badge>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <p className="text-sm text-muted-foreground">Teams</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {orgData.teams?.map((team, index) => (
                    <Badge key={index} variant="secondary">
                      {teamAliasMap[team.team_id] || team.team_id}
                    </Badge>
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

        <TabsContent value="members" className="pt-4">
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
              roleColumnTitle="Organization Role"
              extraColumns={orgExtraColumns}
              emptyText="No members found"
            />
          </div>
        </TabsContent>

        <TabsContent value="settings" className="pt-4">
          <Card className="max-h-[65vh] overflow-y-auto">
            <CardContent>
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-foreground">Organization Settings</h2>
                {canEditOrg && !isEditing && <Button onClick={() => setIsEditing(true)}>Edit Settings</Button>}
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
                    <p className="font-medium text-foreground">Organization Name</p>
                    <div>{orgData.organization_alias}</div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">Organization ID</p>
                    <div className="font-mono">{orgData.organization_id}</div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">Created At</p>
                    <div>{new Date(orgData.created_at).toLocaleString()}</div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">Models</p>
                    <div className="mt-1 flex flex-wrap gap-2">
                      {orgData.models.map((model, index) => (
                        <Badge key={index} variant="secondary">
                          {model}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">Rate Limits</p>
                    <div>TPM: {orgData.litellm_budget_table.tpm_limit || "Unlimited"}</div>
                    <div>RPM: {orgData.litellm_budget_table.rpm_limit || "Unlimited"}</div>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">Budget</p>
                    <div>
                      Max:{" "}
                      {orgData.litellm_budget_table.max_budget !== null
                        ? `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`
                        : "No Limit"}
                    </div>
                    <div>Reset: {orgData.litellm_budget_table.budget_duration || "Never"}</div>
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
        title="Add Organization Member"
        roles={[
          {
            label: "org_admin",
            value: "org_admin",
            description: "Can add and remove members, and change their roles.",
          },
          {
            label: "internal_user",
            value: "internal_user",
            description: "Can view/create keys for themselves within organization.",
          },
          {
            label: "internal_user_viewer",
            value: "internal_user_viewer",
            description: "Can only view their keys within organization.",
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
          title: "Edit Member",
          showEmail: true,
          showUserId: true,
          roleOptions: [
            { label: "Org Admin", value: "org_admin" },
            { label: "Internal User", value: "internal_user" },
            { label: "Internal User Viewer", value: "internal_user_viewer" },
          ],
        }}
      />
    </div>
  );
};

export default OrganizationInfoView;
