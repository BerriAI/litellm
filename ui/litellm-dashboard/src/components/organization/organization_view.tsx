import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import {
  organizationKeys,
  useOrganization,
} from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useQueryClient } from "@tanstack/react-query";
import {
  formatNumberWithCommas,
  copyToClipboard as utilCopyToClipboard,
} from "@/utils/dataUtils";
import { createTeamAliasMap } from "@/utils/teamUtils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, CheckIcon, CopyIcon } from "lucide-react";
import React, { useMemo, useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import MemberTable from "../common_components/MemberTable";
import UserSearchModal from "../common_components/user_search_modal";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import { ModelSelect } from "../ModelSelect/ModelSelect";
import NotificationsManager from "../molecules/notifications_manager";
import {
  Member,
  organizationMemberAddCall,
  organizationMemberDeleteCall,
  organizationMemberUpdateCall,
  organizationUpdateCall,
} from "../networking";
import ObjectPermissionsView from "../object_permissions_view";
import NumericalInput from "../shared/numerical_input";
import MemberModal from "../team/EditMembership";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";

interface OrganizationInfoProps {
  organizationId: string;
  onClose: () => void;
  accessToken: string | null;
  is_org_admin: boolean;
  is_proxy_admin: boolean;
  userModels: string[];
  editOrg: boolean;
}

type OrgEditValues = {
  organization_alias: string;
  models: string[];
  tpm_limit: number | null;
  rpm_limit: number | null;
  max_budget: number | null;
  budget_duration: string | null;
  metadata: string;
  vector_stores: string[];
  mcp_servers_and_groups: { servers: string[]; accessGroups: string[] };
};

// Member-table extra columns rendered as plain JSX. MemberTable's
// MemberTableExtraColumn type matches this shape structurally.
type MemberExtraColumn = {
  title: string;
  key: string;
  render: (_: unknown, record: Member) => React.ReactNode;
};

const OrganizationInfoView: React.FC<OrganizationInfoProps> = ({
  organizationId,
  onClose,
  accessToken,
  is_org_admin,
  is_proxy_admin,
  userModels: _userModels,
  editOrg,
}) => {
  const queryClient = useQueryClient();
  const { data: orgData, isLoading: loading } = useOrganization(organizationId);
  const [isEditing, setIsEditing] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(
    null,
  );
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [isOrgSaving, setIsOrgSaving] = useState(false);
  const canEditOrg = is_org_admin || is_proxy_admin;
  const { data: teams } = useTeams();
  const teamAliasMap = useMemo(() => createTeamAliasMap(teams), [teams]);

  const editForm = useForm<OrgEditValues>({
    defaultValues: {
      organization_alias: "",
      models: [],
      tpm_limit: null,
      rpm_limit: null,
      max_budget: null,
      budget_duration: null,
      metadata: "",
      vector_stores: [],
      mcp_servers_and_groups: { servers: [], accessGroups: [] },
    },
    mode: "onSubmit",
  });

  // Reset form when entering edit mode (or when orgData arrives).
  React.useEffect(() => {
    if (orgData && isEditing) {
      editForm.reset({
        organization_alias: orgData.organization_alias,
        models: orgData.models,
        tpm_limit: orgData.litellm_budget_table.tpm_limit,
        rpm_limit: orgData.litellm_budget_table.rpm_limit,
        max_budget: orgData.litellm_budget_table.max_budget,
        budget_duration: orgData.litellm_budget_table.budget_duration ?? null,
        metadata: orgData.metadata
          ? JSON.stringify(orgData.metadata, null, 2)
          : "",
        vector_stores: orgData.object_permission?.vector_stores || [],
        mcp_servers_and_groups: {
          servers: orgData.object_permission?.mcp_servers || [],
          accessGroups: orgData.object_permission?.mcp_access_groups || [],
        },
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orgData, isEditing]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleMemberAdd = async (values: any) => {
    try {
      if (accessToken == null) return;
      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };
      await organizationMemberAddCall(accessToken, organizationId, member);
      NotificationsManager.success("Organization member added successfully");
      setIsAddMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend("Failed to add organization member");
      console.error("Error adding organization member:", error);
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleMemberUpdate = async (values: any) => {
    try {
      if (!accessToken) return;
      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };
      await organizationMemberUpdateCall(accessToken, organizationId, member);
      NotificationsManager.success("Organization member updated successfully");
      setIsEditMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update organization member");
      console.error("Error updating organization member:", error);
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleMemberDelete = async (values: any) => {
    try {
      if (!accessToken) return;
      await organizationMemberDeleteCall(
        accessToken,
        organizationId,
        values.user_id,
      );
      NotificationsManager.success("Organization member deleted successfully");
      setIsEditMemberModalVisible(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend("Failed to delete organization member");
      console.error("Error deleting organization member:", error);
    }
  };

  const handleOrgUpdate = editForm.handleSubmit(async (values) => {
    try {
      if (!accessToken || !orgData) return;
      setIsOrgSaving(true);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const updateData: any = {
        organization_id: organizationId,
        organization_alias: values.organization_alias,
        models: values.models,
        litellm_budget_table: {
          tpm_limit: values.tpm_limit,
          rpm_limit: values.rpm_limit,
          max_budget: values.max_budget,
          budget_duration: values.budget_duration,
        },
        metadata: values.metadata ? JSON.parse(values.metadata) : null,
      };
      if (
        values.vector_stores !== undefined ||
        values.mcp_servers_and_groups !== undefined
      ) {
        updateData.object_permission = {
          ...orgData?.object_permission,
          vector_stores: values.vector_stores || [],
        };
        const { servers, accessGroups } = values.mcp_servers_and_groups || {
          servers: [],
          accessGroups: [],
        };
        if (servers && servers.length > 0)
          updateData.object_permission.mcp_servers = servers;
        if (accessGroups && accessGroups.length > 0)
          updateData.object_permission.mcp_access_groups = accessGroups;
      }
      await organizationUpdateCall(accessToken, updateData);
      NotificationsManager.success("Organization settings updated successfully");
      setIsEditing(false);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update organization settings");
      console.error("Error updating organization:", error);
    } finally {
      setIsOrgSaving(false);
    }
  });

  if (loading) return <div className="p-4">Loading...</div>;
  if (!orgData) return <div className="p-4">Organization not found</div>;

  const copyToClipboard = async (
    text: string | null | undefined,
    key: string,
  ) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(
        () => setCopiedStates((prev) => ({ ...prev, [key]: false })),
        2000,
      );
    }
  };

  const orgExtraColumns: MemberExtraColumn[] = [
    {
      title: "Spend (USD)",
      key: "spend",
      render: (_: unknown, record: Member) => {
        const orgMember =
          record.user_id != null
            ? (orgData.members || []).find((m) => m.user_id === record.user_id)
            : undefined;
        return (
          <span>${formatNumberWithCommas(orgMember?.spend ?? 0, 4)}</span>
        );
      },
    },
    {
      title: "Created At",
      key: "created_at",
      render: (_: unknown, record: Member) => {
        const orgMember =
          record.user_id != null
            ? (orgData.members || []).find((m) => m.user_id === record.user_id)
            : undefined;
        return (
          <span>
            {orgMember?.created_at
              ? new Date(orgMember.created_at).toLocaleString()
              : "-"}
          </span>
        );
      },
    },
  ];

  return (
    <div className="w-full h-screen p-4 bg-background">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button variant="ghost" size="sm" onClick={onClose} className="mb-4">
            <ArrowLeft className="h-4 w-4" />
            Back to Organizations
          </Button>
          <h2 className="text-2xl font-semibold">{orgData.organization_alias}</h2>
          <div className="flex items-center cursor-pointer">
            <span className="text-muted-foreground font-mono">
              {orgData.organization_id}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                copyToClipboard(orgData.organization_id, "org-id")
              }
              className="ml-1"
              aria-label="Copy organization id"
            >
              {copiedStates["org-id"] ? (
                <CheckIcon size={12} />
              ) : (
                <CopyIcon size={12} />
              )}
            </Button>
          </div>
        </div>
      </div>

      <Tabs
        defaultValue={editOrg ? "settings" : "overview"}
        className="mb-4"
      >
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="members">Members</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="p-4">
              <p className="text-sm font-medium">Organization Details</p>
              <div className="mt-2 text-sm space-y-1">
                <p>Created: {new Date(orgData.created_at).toLocaleDateString()}</p>
                <p>Updated: {new Date(orgData.updated_at).toLocaleDateString()}</p>
                <p>Created By: {orgData.created_by}</p>
              </div>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium">Budget Status</p>
              <div className="mt-2">
                <h3 className="text-xl font-semibold">
                  ${formatNumberWithCommas(orgData.spend, 4)}
                </h3>
                <p className="text-sm">
                  of{" "}
                  {orgData.litellm_budget_table.max_budget === null
                    ? "Unlimited"
                    : `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`}
                </p>
                {orgData.litellm_budget_table.budget_duration && (
                  <p className="text-sm text-muted-foreground">
                    Reset: {orgData.litellm_budget_table.budget_duration}
                  </p>
                )}
              </div>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium">Rate Limits</p>
              <div className="mt-2 text-sm space-y-1">
                <p>TPM: {orgData.litellm_budget_table.tpm_limit || "Unlimited"}</p>
                <p>RPM: {orgData.litellm_budget_table.rpm_limit || "Unlimited"}</p>
                {orgData.litellm_budget_table.max_parallel_requests && (
                  <p>
                    Max Parallel Requests:{" "}
                    {orgData.litellm_budget_table.max_parallel_requests}
                  </p>
                )}
              </div>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium">Models</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {orgData.models.length === 0 ? (
                  <Badge variant="destructive">All proxy models</Badge>
                ) : (
                  orgData.models.map((model, index) => (
                    <Badge key={index} variant="destructive">
                      {model}
                    </Badge>
                  ))
                )}
              </div>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium">Teams</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {orgData.teams?.map((team, index) => (
                  <Badge key={index} variant="destructive">
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
          </div>
        </TabsContent>

        <TabsContent value="members">
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

        <TabsContent value="settings">
          <Card className="overflow-y-auto max-h-[65vh] p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Organization Settings</h3>
              {canEditOrg && !isEditing && (
                <Button onClick={() => setIsEditing(true)}>
                  Edit Settings
                </Button>
              )}
            </div>

            {isEditing ? (
              <FormProvider {...editForm}>
                <form onSubmit={handleOrgUpdate} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="org_alias">
                      Organization Name <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      id="org_alias"
                      {...editForm.register("organization_alias", {
                        required: "Please input an organization name",
                      })}
                    />
                    {editForm.formState.errors.organization_alias && (
                      <p className="text-sm text-destructive">
                        {editForm.formState.errors.organization_alias.message as string}
                      </p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label>Models</Label>
                    <Controller
                      control={editForm.control}
                      name="models"
                      render={({ field }) => (
                        <ModelSelect
                          value={field.value}
                          onChange={(values) => field.onChange(values)}
                          context="organization"
                          options={{
                            includeSpecialOptions: true,
                            showAllProxyModelsOverride: true,
                          }}
                        />
                      )}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="max_budget">Max Budget (USD)</Label>
                    <Controller
                      control={editForm.control}
                      name="max_budget"
                      render={({ field }) => (
                        <NumericalInput
                          step={0.01}
                          precision={2}
                          style={{ width: "100%" }}
                          value={field.value ?? undefined}
                          onChange={(v: number | null | undefined) =>
                            field.onChange(v ?? null)
                          }
                        />
                      )}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Reset Budget</Label>
                    <Controller
                      control={editForm.control}
                      name="budget_duration"
                      render={({ field }) => (
                        <Select
                          value={field.value ?? ""}
                          onValueChange={(v) => field.onChange(v || null)}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="n/a" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="24h">daily</SelectItem>
                            <SelectItem value="7d">weekly</SelectItem>
                            <SelectItem value="30d">monthly</SelectItem>
                          </SelectContent>
                        </Select>
                      )}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Tokens per minute Limit (TPM)</Label>
                    <Controller
                      control={editForm.control}
                      name="tpm_limit"
                      render={({ field }) => (
                        <NumericalInput
                          step={1}
                          style={{ width: "100%" }}
                          value={field.value ?? undefined}
                          onChange={(v: number | null | undefined) =>
                            field.onChange(v ?? null)
                          }
                        />
                      )}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Requests per minute Limit (RPM)</Label>
                    <Controller
                      control={editForm.control}
                      name="rpm_limit"
                      render={({ field }) => (
                        <NumericalInput
                          step={1}
                          style={{ width: "100%" }}
                          value={field.value ?? undefined}
                          onChange={(v: number | null | undefined) =>
                            field.onChange(v ?? null)
                          }
                        />
                      )}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Vector Stores</Label>
                    <Controller
                      control={editForm.control}
                      name="vector_stores"
                      render={({ field }) => (
                        <VectorStoreSelector
                          onChange={(values) => field.onChange(values)}
                          value={field.value}
                          accessToken={accessToken || ""}
                          placeholder="Select vector stores"
                        />
                      )}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>MCP Servers & Access Groups</Label>
                    <Controller
                      control={editForm.control}
                      name="mcp_servers_and_groups"
                      render={({ field }) => (
                        <MCPServerSelector
                          onChange={(values) => field.onChange(values)}
                          value={field.value}
                          accessToken={accessToken || ""}
                          placeholder="Select MCP servers and access groups"
                        />
                      )}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="metadata">Metadata</Label>
                    <Textarea
                      id="metadata"
                      rows={4}
                      {...editForm.register("metadata")}
                    />
                  </div>

                  <div className="sticky z-10 bg-background p-4 border-t border-border bottom-[-1.5rem] inset-x-[-1.5rem]">
                    <div className="flex justify-end items-center gap-2">
                      <Button
                        type="button"
                        variant="secondary"
                        onClick={() => setIsEditing(false)}
                        disabled={isOrgSaving}
                      >
                        Cancel
                      </Button>
                      <Button type="submit" disabled={isOrgSaving}>
                        {isOrgSaving ? "Saving…" : "Save Changes"}
                      </Button>
                    </div>
                  </div>
                </form>
              </FormProvider>
            ) : (
              <div className="space-y-4 text-sm">
                <div>
                  <p className="font-medium">Organization Name</p>
                  <div>{orgData.organization_alias}</div>
                </div>
                <div>
                  <p className="font-medium">Organization ID</p>
                  <div className="font-mono">{orgData.organization_id}</div>
                </div>
                <div>
                  <p className="font-medium">Created At</p>
                  <div>{new Date(orgData.created_at).toLocaleString()}</div>
                </div>
                <div>
                  <p className="font-medium">Models</p>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {orgData.models.map((model, index) => (
                      <Badge key={index} variant="destructive">
                        {model}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="font-medium">Rate Limits</p>
                  <div>
                    TPM: {orgData.litellm_budget_table.tpm_limit || "Unlimited"}
                  </div>
                  <div>
                    RPM: {orgData.litellm_budget_table.rpm_limit || "Unlimited"}
                  </div>
                </div>
                <div>
                  <p className="font-medium">Budget</p>
                  <div>
                    Max:{" "}
                    {orgData.litellm_budget_table.max_budget !== null
                      ? `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`
                      : "No Limit"}
                  </div>
                  <div>
                    Reset: {orgData.litellm_budget_table.budget_duration || "Never"}
                  </div>
                </div>

                <ObjectPermissionsView
                  objectPermission={orgData.object_permission}
                  variant="inline"
                  className="pt-4 border-t border-border"
                  accessToken={accessToken}
                />
              </div>
            )}
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
            description:
              "Can view/create keys for themselves within organization.",
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
            {
              label: "Internal User Viewer",
              value: "internal_user_viewer",
            },
          ],
        }}
      />
    </div>
  );
};

export default OrganizationInfoView;
