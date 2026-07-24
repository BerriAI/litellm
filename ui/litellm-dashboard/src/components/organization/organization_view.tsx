import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { organizationKeys, useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useQueryClient } from "@tanstack/react-query";
import { MoneyCell } from "@/components/shared/table_cells";
import { formatNumberWithCommas, copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { createTeamAliasMap } from "@/utils/teamUtils";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { Badge, Card, Grid, Text, Title, Button as TremorButton } from "@tremor/react";
import { Button, Tabs, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { CheckIcon, CopyIcon } from "lucide-react";
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
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
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

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

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
        return (
          <Typography.Text>
            {orgMember?.created_at ? new Date(orgMember.created_at).toLocaleString() : "-"}
          </Typography.Text>
        );
      },
    },
  ];

  return (
    <div className="w-full h-screen p-4 bg-white">
      <div className="flex justify-between items-center mb-6">
        <div>
          <TremorButton icon={ArrowLeftIcon} onClick={onClose} variant="light" className="mb-4">
            Back to Organizations
          </TremorButton>
          <Title>{orgData.organization_alias}</Title>
          <div className="flex items-center cursor-pointer">
            <Text className="text-gray-500 font-mono">{orgData.organization_id}</Text>
            <Button
              type="text"
              size="small"
              icon={copiedStates["org-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(orgData.organization_id, "org-id")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["org-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
        </div>
      </div>

      <Tabs
        defaultActiveKey={editOrg ? "settings" : "overview"}
        className="mb-4"
        items={[
          {
            key: "overview",
            label: "Overview",
            children: (
              <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
                <Card>
                  <Text>Organization Details</Text>
                  <div className="mt-2">
                    <Text>Created: {new Date(orgData.created_at).toLocaleDateString()}</Text>
                    <Text>Updated: {new Date(orgData.updated_at).toLocaleDateString()}</Text>
                    <Text>Created By: {orgData.created_by}</Text>
                  </div>
                </Card>

                <Card>
                  <Text>Budget Status</Text>
                  <div className="mt-2">
                    <Title>${formatNumberWithCommas(orgData.spend, 4)}</Title>
                    <Text>
                      of{" "}
                      {orgData.litellm_budget_table.max_budget === null
                        ? "Unlimited"
                        : `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`}
                    </Text>
                    {orgData.litellm_budget_table.budget_duration && (
                      <Text className="text-gray-500">Reset: {orgData.litellm_budget_table.budget_duration}</Text>
                    )}
                  </div>
                </Card>

                <Card>
                  <Text>Rate Limits</Text>
                  <div className="mt-2">
                    <Text>TPM: {orgData.litellm_budget_table.tpm_limit || "Unlimited"}</Text>
                    <Text>RPM: {orgData.litellm_budget_table.rpm_limit || "Unlimited"}</Text>
                    {orgData.litellm_budget_table.max_parallel_requests && (
                      <Text>Max Parallel Requests: {orgData.litellm_budget_table.max_parallel_requests}</Text>
                    )}
                  </div>
                </Card>

                <Card>
                  <Text>Models</Text>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {orgData.models.length === 0 ? (
                      <Badge color="red">All proxy models</Badge>
                    ) : (
                      orgData.models.map((model, index) => (
                        <Badge key={index} color="red">
                          {model}
                        </Badge>
                      ))
                    )}
                  </div>
                </Card>
                <Card>
                  <Text>Teams</Text>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {orgData.teams?.map((team, index) => (
                      <Badge key={index} color="red">
                        {teamAliasMap[team.team_id] || team.team_id}
                      </Badge>
                    ))}
                  </div>
                </Card>

                <ObjectPermissionsView
                  objectPermission={orgData.object_permission}
                  variant="card"
                  accessToken={accessToken}
                />
              </Grid>
            ),
          },
          {
            key: "members",
            label: "Members",
            children: (
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
            ),
          },
          {
            key: "settings",
            label: "Settings",
            children: (
              <Card className="overflow-y-auto max-h-[65vh]">
                <div className="flex justify-between items-center mb-4">
                  <Title>Organization Settings</Title>
                  {canEditOrg && !isEditing && (
                    <TremorButton onClick={() => setIsEditing(true)}>Edit Settings</TremorButton>
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
                  <div className="space-y-4">
                    <div>
                      <Text className="font-medium">Organization Name</Text>
                      <div>{orgData.organization_alias}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Organization ID</Text>
                      <div className="font-mono">{orgData.organization_id}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Created At</Text>
                      <div>{new Date(orgData.created_at).toLocaleString()}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Models</Text>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {orgData.models.map((model, index) => (
                          <Badge key={index} color="red">
                            {model}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <Text className="font-medium">Rate Limits</Text>
                      <div>TPM: {orgData.litellm_budget_table.tpm_limit || "Unlimited"}</div>
                      <div>RPM: {orgData.litellm_budget_table.rpm_limit || "Unlimited"}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Budget</Text>
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
                      className="pt-4 border-t border-gray-200"
                      accessToken={accessToken}
                    />
                  </div>
                )}
              </Card>
            ),
          },
        ]}
      />
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
