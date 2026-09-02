import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  userGetInfoV2,
  UserInfoV2Response,
  userDeleteCall,
  userUpdateUserCall,
  modelAvailableCall,
  invitationCreateCall,
  getProxyBaseUrl,
  teamInfoCall,
  teamListCall,
  teamMemberAddCall,
  teamMemberDeleteCall,
  Member,
} from "@/components/networking";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { rolesWithWriteAccess } from "@/utils/roles";
import { teamDetailHref } from "@/utils/entityLinks";
import { BadgeLink } from "@/components/shared/BadgeLink";
import { UserEditView } from "../user_edit_view";
import OnboardingModal, { InvitationLink } from "@/components/onboarding_link";
import { formatNumberWithCommas, copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { ArrowLeft, CheckIcon, CopyIcon, Plus, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "@/lib/toast";
import { getBudgetDurationLabel } from "@/components/common_components/budget_duration_dropdown";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import MCPServerPermissions from "@/components/permissions/MCPServerPermissions";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { extractMcpEntitlement } from "@/components/mcp_server_management/mcpEntitlement";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface UserInfoViewProps {
  userId: string;
  onClose: () => void;
  accessToken: string | null;
  userRole: string | null;
  onDelete?: () => void;
  possibleUIRoles: Record<string, Record<string, string>> | null;
  initialTab?: number; // 0 for Overview, 1 for Details
  startInEditMode?: boolean;
}

/** Team info used for display in user detail view */
interface TeamDisplayInfo {
  team_id: string;
  team_alias: string | null;
}

const ADD_TEAM_FIELD_ID = "add-team-team";
const ADD_TEAM_ROLE_FIELD_ID = "add-team-role";

interface TeamOption {
  team_id: string;
  team_alias: string;
}

const MEMBER_ROLE_OPTIONS = [
  { value: "user", hint: "Can view team info, but not manage it" },
  { value: "admin", hint: "Can create team keys, add members, and manage settings" },
] as const;

export default function UserInfoView({
  userId,
  onClose,
  accessToken,
  userRole,
  onDelete,
  possibleUIRoles,
  initialTab = 0,
  startInEditMode = false,
}: UserInfoViewProps) {
  const { premiumUser } = useAuthorized();
  const [userData, setUserData] = useState<UserInfoV2Response | null>(null);
  const [teamDetails, setTeamDetails] = useState<TeamDisplayInfo[]>([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeletingUser, setIsDeletingUser] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(startInEditMode);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] = useState(false);
  const [invitationLinkData, setInvitationLinkData] = useState<InvitationLink | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>(initialTab === 1 ? "details" : "overview");
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [isTeamsExpanded, setIsTeamsExpanded] = useState(false);
  const [isAddTeamModalOpen, setIsAddTeamModalOpen] = useState(false);
  const [isRemoveTeamModalOpen, setIsRemoveTeamModalOpen] = useState(false);
  const [teamToRemove, setTeamToRemove] = useState<TeamDisplayInfo | null>(null);
  const [isAddingTeam, setIsAddingTeam] = useState(false);
  const [isRemovingTeam, setIsRemovingTeam] = useState(false);
  const [allTeams, setAllTeams] = useState<Array<{ team_id: string; team_alias: string }>>([]);
  const [selectedTeamId, setSelectedTeamId] = useState<string>("");
  const [selectedRole, setSelectedRole] = useState<string>("user");
  const [isLoadingTeams, setIsLoadingTeams] = useState(false);
  const { data: allMcpServers = [] } = useMCPServers();
  const { data: allMcpToolsets = [] } = useMCPToolsets();

  React.useEffect(() => {
    setBaseUrl(getProxyBaseUrl());
  }, []);

  React.useEffect(() => {
    const fetchData = async () => {
      try {
        if (!accessToken) return;
        const data = await userGetInfoV2(accessToken, userId);
        setUserData(data);

        // Fetch team details for display (team aliases)
        if (data.teams && data.teams.length > 0) {
          try {
            const teamPromises = data.teams.map(async (teamId: string) => {
              try {
                const teamData = await teamInfoCall(accessToken, teamId);
                return {
                  team_id: teamId,
                  team_alias: teamData?.team_info?.team_alias || null,
                };
              } catch {
                return { team_id: teamId, team_alias: null };
              }
            });
            const teams = await Promise.all(teamPromises);
            setTeamDetails(teams);
          } catch {
            // Fall back to just team IDs
            setTeamDetails(data.teams.map((id: string) => ({ team_id: id, team_alias: null })));
          }
        }

        // Fetch available models
        const modelDataResponse = await modelAvailableCall(accessToken, userId, userRole || "");
        const availableModels = modelDataResponse.data.map((model: any) => model.id);
        setUserModels(availableModels);
      } catch (error) {
        console.error("Error fetching user data:", error);
        toast.fromError("Failed to fetch user data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [accessToken, userId, userRole]);

  const isProxyAdmin = userRole === "proxy_admin" || userRole === "Admin";

  const fetchAllTeams = async () => {
    if (!accessToken) return;
    setIsLoadingTeams(true);
    try {
      const teams = await teamListCall(accessToken, null);
      setAllTeams(
        (teams || []).map((t: any) => ({
          team_id: t.team_id,
          team_alias: t.team_alias || t.team_id,
        })),
      );
    } catch (error) {
      console.error("Error fetching teams:", error);
    } finally {
      setIsLoadingTeams(false);
    }
  };

  const handleOpenAddTeamModal = () => {
    setSelectedTeamId("");
    setSelectedRole("user");
    setIsAddTeamModalOpen(true);
    fetchAllTeams();
  };

  const handleAddTeamSubmit = async () => {
    if (!accessToken || !selectedTeamId) return;
    setIsAddingTeam(true);
    try {
      const member: Member = {
        role: selectedRole,
        user_id: userId,
      };
      await teamMemberAddCall(accessToken, selectedTeamId, member);
      toast.success("User added to team successfully");
      setIsAddTeamModalOpen(false);
      // Re-fetch user data to refresh teams
      const data = await userGetInfoV2(accessToken, userId);
      setUserData(data);
      if (data.teams && data.teams.length > 0) {
        const teamPromises = data.teams.map(async (teamId: string) => {
          try {
            const teamData = await teamInfoCall(accessToken, teamId);
            return { team_id: teamId, team_alias: teamData?.team_info?.team_alias || null };
          } catch {
            return { team_id: teamId, team_alias: null };
          }
        });
        setTeamDetails(await Promise.all(teamPromises));
      } else {
        setTeamDetails([]);
      }
    } catch (error: any) {
      console.error("Error adding user to team:", error);
      toast.fromError(error?.message || "Failed to add user to team");
    } finally {
      setIsAddingTeam(false);
    }
  };

  const handleOpenRemoveTeamModal = (team: TeamDisplayInfo) => {
    setTeamToRemove(team);
    setIsRemoveTeamModalOpen(true);
  };

  const handleRemoveTeamConfirm = async () => {
    if (!accessToken || !teamToRemove) return;
    setIsRemovingTeam(true);
    try {
      const member: Member = {
        role: "user",
        user_id: userId,
      };
      await teamMemberDeleteCall(accessToken, teamToRemove.team_id, member);
      toast.success("User removed from team successfully");
      setIsRemoveTeamModalOpen(false);
      setTeamToRemove(null);
      // Re-fetch user data to refresh teams
      const data = await userGetInfoV2(accessToken, userId);
      setUserData(data);
      if (data.teams && data.teams.length > 0) {
        const teamPromises = data.teams.map(async (teamId: string) => {
          try {
            const teamData = await teamInfoCall(accessToken, teamId);
            return { team_id: teamId, team_alias: teamData?.team_info?.team_alias || null };
          } catch {
            return { team_id: teamId, team_alias: null };
          }
        });
        setTeamDetails(await Promise.all(teamPromises));
      } else {
        setTeamDetails([]);
      }
    } catch (error: any) {
      console.error("Error removing user from team:", error);
      toast.fromError(error?.message || "Failed to remove user from team");
    } finally {
      setIsRemovingTeam(false);
    }
  };

  const handleRemoveTeamCancel = () => {
    setIsRemoveTeamModalOpen(false);
    setTeamToRemove(null);
  };

  const availableTeamsForAdd = allTeams.filter((t) => !teamDetails.some((td) => td.team_id === t.team_id));

  const selectedTeamOption = availableTeamsForAdd.find((team) => team.team_id === selectedTeamId) ?? null;

  const handleResetPassword = async () => {
    if (!accessToken) {
      toast.fromError("Access token not found");
      return;
    }
    try {
      toast.success("Generating password reset link...");
      const data = await invitationCreateCall(accessToken, userId);
      setInvitationLinkData(data);
      setIsInvitationLinkModalVisible(true);
    } catch (error) {
      toast.fromError("Failed to generate password reset link");
    }
  };

  const handleDelete = async () => {
    try {
      if (!accessToken) return;
      setIsDeletingUser(true);
      await userDeleteCall(accessToken, [userId]);
      toast.success("User deleted successfully");
      if (onDelete) {
        onDelete();
      }
      onClose();
    } catch (error) {
      console.error("Error deleting user:", error);
      toast.fromError("Failed to delete user");
    } finally {
      setIsDeleteModalOpen(false);
      setIsDeletingUser(false);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
  };

  const handleUserUpdate = async (formValues: Record<string, any>) => {
    try {
      if (!accessToken || !userData) return;

      const mcpEntitlement = extractMcpEntitlement(formValues, allMcpServers, allMcpToolsets);
      const userFields = Object.fromEntries(
        Object.entries(formValues).filter(
          ([field]) => field !== "mcp_servers_and_groups" && field !== "mcp_tool_permissions",
        ),
      );

      await userUpdateUserCall(
        accessToken,
        mcpEntitlement ? { ...userFields, object_permission: mcpEntitlement } : userFields,
        null,
      );

      // Update local state with new values
      setUserData({
        ...userData,
        user_email: formValues.user_email ?? userData.user_email,
        user_alias: formValues.user_alias ?? userData.user_alias,
        models: formValues.models ?? userData.models,
        max_budget: formValues.max_budget ?? userData.max_budget,
        budget_duration: formValues.budget_duration ?? userData.budget_duration,
        metadata: formValues.metadata ?? userData.metadata,
        model_max_budget: formValues.model_max_budget ?? userData.model_max_budget,
        object_permission: mcpEntitlement
          ? { ...userData.object_permission, ...mcpEntitlement }
          : userData.object_permission,
      });

      toast.success("User updated successfully");
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating user:", error);
      toast.fromError("Failed to update user");
    }
  };

  if (isLoading) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeft />
          Back to Users
        </Button>
        <p className="text-sm">Loading user data...</p>
      </div>
    );
  }

  if (!userData) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeft />
          Back to Users
        </Button>
        <p className="text-sm">User not found</p>
      </div>
    );
  }

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  // Build a legacy-compatible shape for UserEditView
  const userDataForEdit = {
    user_id: userData.user_id,
    user_info: {
      user_email: userData.user_email,
      user_alias: userData.user_alias,
      user_role: userData.user_role,
      models: userData.models,
      max_budget: userData.max_budget,
      budget_duration: userData.budget_duration,
      metadata: userData.metadata,
      // Without these the per-model budget editor mounts empty and a save
      // replaces the user's existing budgets with whatever was typed.
      model_max_budget: userData.model_max_budget,
      model_max_budget_usage: userData.model_max_budget_usage,
    },
  };

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button variant="ghost" onClick={onClose} className="mb-4">
            <ArrowLeft />
            Back to Users
          </Button>
          <h2 className="text-xl font-semibold">{userData.user_email || "User"}</h2>
          <div className="flex items-center cursor-pointer">
            <span className="text-sm text-muted-foreground font-mono">{userData.user_id}</span>
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => copyToClipboard(userData.user_id, "user-id")}
              className={`left-2 z-raised transition-all duration-200 ${
                copiedStates["user-id"]
                  ? "text-success bg-success/10 border-success/20"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              }`}
            >
              {copiedStates["user-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
            </Button>
          </div>
        </div>
        {userRole && rolesWithWriteAccess.includes(userRole) && (
          <div className="flex items-center space-x-2">
            <Button variant="secondary" onClick={handleResetPassword} className="flex items-center">
              <RefreshCw />
              Reset Password
            </Button>
            <Button
              variant="secondary"
              onClick={() => setIsDeleteModalOpen(true)}
              className="flex items-center text-destructive border-destructive hover:bg-destructive/10"
            >
              <Trash2 />
              Delete User
            </Button>
          </div>
        )}
      </div>

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete User?"
        message="Are you sure you want to delete this user? This action cannot be undone."
        resourceInformationTitle="User Information"
        resourceInformation={[
          { label: "Email", value: userData.user_email },
          { label: "User ID", value: userData.user_id, code: true },
          {
            label: "Global Proxy Role",
            value: (userData.user_role && possibleUIRoles?.[userData.user_role]?.ui_label) || userData.user_role || "-",
          },
          {
            label: "Total Spend (USD)",
            value: userData.spend !== null && userData.spend !== undefined ? userData.spend.toFixed(2) : undefined,
          },
        ]}
        onCancel={cancelDelete}
        onOk={handleDelete}
        confirmLoading={isDeletingUser}
      />

      <Tabs value={activeTab} onValueChange={(v: unknown) => setActiveTab(String(v))} className="gap-0">
        <TabsList variant="line" className="mb-4">
          <TabsTrigger value="overview" className="flex-none data-active:text-primary after:bg-primary">
            Overview
          </TabsTrigger>
          <TabsTrigger value="details" className="flex-none data-active:text-primary after:bg-primary">
            Details
          </TabsTrigger>
        </TabsList>

        {/* Overview Panel */}
        <TabsContent value="overview" keepMounted>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="block p-6">
              <p>Spend</p>
              <div className="mt-2">
                <h3 className="text-lg font-medium">${formatNumberWithCommas(userData.spend || 0, 2)}</h3>
                <p>
                  of {userData.max_budget !== null ? `$${formatNumberWithCommas(userData.max_budget, 2)}` : "Unlimited"}
                </p>
              </div>
            </Card>

            <Card className="block p-6">
              <div className="flex justify-between items-center mb-2">
                <p>Teams</p>
                {isProxyAdmin && (
                  <Button variant="ghost" size="sm" onClick={handleOpenAddTeamModal}>
                    <Plus />
                    Add Team
                  </Button>
                )}
              </div>
              <div className="mt-2">
                {teamDetails.length > 0 ? (
                  <div className="max-h-60 overflow-y-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Team Name</TableHead>
                          {isProxyAdmin && <TableHead className="text-right">Actions</TableHead>}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {teamDetails.slice(0, isTeamsExpanded ? teamDetails.length : 20).map((team) => (
                          <TableRow key={team.team_id}>
                            <TableCell>
                              <BadgeLink href={teamDetailHref(team.team_id)}>
                                {team.team_alias || team.team_id}
                              </BadgeLink>
                            </TableCell>
                            {isProxyAdmin && (
                              <TableCell className="text-right">
                                <Button
                                  variant="ghost"
                                  size="icon-sm"
                                  aria-label={`Remove from ${team.team_alias || team.team_id}`}
                                  onClick={() => handleOpenRemoveTeamModal(team)}
                                  className="text-destructive"
                                >
                                  <Trash2 />
                                </Button>
                              </TableCell>
                            )}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <p>No teams</p>
                )}
                {!isTeamsExpanded && teamDetails.length > 20 && (
                  <Button variant="ghost" size="sm" className="mt-2" onClick={() => setIsTeamsExpanded(true)}>
                    +{teamDetails.length - 20} more
                  </Button>
                )}
                {isTeamsExpanded && teamDetails.length > 20 && (
                  <Button variant="ghost" size="sm" className="mt-2" onClick={() => setIsTeamsExpanded(false)}>
                    Show Less
                  </Button>
                )}
              </div>
            </Card>

            <Card className="block p-6">
              <p>Personal Models</p>
              <div className="mt-2">
                {userData.models?.length && userData.models?.length > 0 ? (
                  userData.models?.map((model, index) => <p key={index}>{model}</p>)
                ) : (
                  <p>All proxy models</p>
                )}
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Details Panel */}
        <TabsContent value="details" keepMounted>
          <Card className="block p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium">User Settings</h3>
              {!isEditing && userRole && rolesWithWriteAccess.includes(userRole) && (
                <Button onClick={() => setIsEditing(true)}>Edit Settings</Button>
              )}
            </div>

            {isEditing && userData ? (
              <UserEditView
                userData={userDataForEdit}
                onCancel={() => setIsEditing(false)}
                onSubmit={handleUserUpdate}
                teams={teamDetails}
                accessToken={accessToken}
                userID={userId}
                userRole={userRole}
                userModels={userModels}
                possibleUIRoles={possibleUIRoles}
                objectPermission={userData.object_permission}
                premiumUser={premiumUser === true}
              />
            ) : (
              <div className="space-y-4">
                <div>
                  <p className="font-medium">User ID</p>
                  <div className="flex items-center cursor-pointer">
                    <span className="font-mono">{userData.user_id}</span>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      onClick={() => copyToClipboard(userData.user_id, "user-id")}
                      className={`left-2 z-raised transition-all duration-200 ${
                        copiedStates["user-id"]
                          ? "text-success bg-success/10 border-success/20"
                          : "text-muted-foreground hover:text-foreground hover:bg-accent"
                      }`}
                    >
                      {copiedStates["user-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
                    </Button>
                  </div>
                </div>

                <div>
                  <p className="font-medium">Email</p>
                  <p>{userData.user_email || "Not Set"}</p>
                </div>

                <div>
                  <p className="font-medium">User Alias</p>
                  <p>{userData.user_alias || "Not Set"}</p>
                </div>

                <div>
                  <p className="font-medium">Global Proxy Role</p>
                  <p>{userData.user_role || "Not Set"}</p>
                </div>

                <div>
                  <p className="font-medium">Created</p>
                  <p>{userData.created_at ? new Date(userData.created_at).toLocaleString() : "Unknown"}</p>
                </div>

                <div>
                  <p className="font-medium">Last Updated</p>
                  <p>{userData.updated_at ? new Date(userData.updated_at).toLocaleString() : "Unknown"}</p>
                </div>

                <div>
                  <p className="font-medium">Personal Models</p>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {userData.models?.length && userData.models?.length > 0 ? (
                      userData.models?.map((model, index) => (
                        <span key={index} className="px-2 py-1 bg-info/15 rounded-sm text-xs">
                          {model}
                        </span>
                      ))
                    ) : (
                      <p>All proxy models</p>
                    )}
                  </div>
                </div>

                <div>
                  <p className="font-medium">Max Budget</p>
                  <p>
                    {userData.max_budget !== null && userData.max_budget !== undefined
                      ? `$${formatNumberWithCommas(userData.max_budget, 4)}`
                      : "Unlimited"}
                  </p>
                </div>

                <div>
                  <p className="font-medium">Budget Reset</p>
                  <p>{getBudgetDurationLabel(userData.budget_duration ?? null)}</p>
                </div>

                <div>
                  <p className="font-medium">Metadata</p>
                  <pre className="bg-muted p-2 rounded-sm text-xs overflow-auto mt-1">
                    {JSON.stringify(userData.metadata || {}, null, 2)}
                  </pre>
                </div>

                <div>
                  <p className="font-medium mb-2">MCP Permissions</p>
                  <MCPServerPermissions
                    mcpServers={userData.object_permission?.mcp_servers || []}
                    mcpAccessGroups={userData.object_permission?.mcp_access_groups || []}
                    mcpToolPermissions={userData.object_permission?.mcp_tool_permissions || {}}
                    mcpToolsets={userData.object_permission?.mcp_toolsets || []}
                    accessToken={accessToken}
                  />
                </div>
              </div>
            )}
          </Card>
        </TabsContent>
      </Tabs>
      <OnboardingModal
        isInvitationLinkModalVisible={isInvitationLinkModalVisible}
        setIsInvitationLinkModalVisible={setIsInvitationLinkModalVisible}
        baseUrl={baseUrl || ""}
        invitationLinkData={invitationLinkData}
        modalType="resetPassword"
      />

      {/* Delete Team Member Modal */}
      <DeleteResourceModal
        isOpen={isRemoveTeamModalOpen}
        title="Remove from Team"
        alertMessage="Removing this user from the team will also delete any keys the user created for this team."
        message="Are you sure you want to remove this user from the team? This action cannot be undone."
        resourceInformationTitle="Team Membership"
        resourceInformation={[
          { label: "Team", value: teamToRemove?.team_alias || teamToRemove?.team_id },
          { label: "User ID", value: userData?.user_id, code: true },
          { label: "Email", value: userData?.user_email },
        ]}
        onCancel={handleRemoveTeamCancel}
        onOk={handleRemoveTeamConfirm}
        confirmLoading={isRemovingTeam}
      />

      {/* Add to Team Modal */}
      <Dialog
        open={isAddTeamModalOpen}
        onOpenChange={(open) => !open && setIsAddTeamModalOpen(false)}
        disablePointerDismissal={isAddingTeam}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Add User to Team</DialogTitle>
          </DialogHeader>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              handleAddTeamSubmit();
            }}
          >
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor={ADD_TEAM_FIELD_ID}>Team</FieldLabel>
                <Combobox
                  items={availableTeamsForAdd}
                  value={selectedTeamOption}
                  onValueChange={(team: TeamOption | null) => setSelectedTeamId(team?.team_id ?? "")}
                  itemToStringLabel={(team: TeamOption) => team.team_alias}
                  isItemEqualToValue={(team: TeamOption, value: TeamOption) => team.team_id === value.team_id}
                >
                  <ComboboxInput id={ADD_TEAM_FIELD_ID} placeholder="Select a team" className="w-full" />
                  <ComboboxContent>
                    <ComboboxEmpty>No teams found</ComboboxEmpty>
                    <ComboboxList>
                      {(team: TeamOption) => (
                        <ComboboxItem key={team.team_id} value={team} title={team.team_alias}>
                          {team.team_alias}
                        </ComboboxItem>
                      )}
                    </ComboboxList>
                  </ComboboxContent>
                </Combobox>
              </Field>

              <Field>
                <FieldLabel htmlFor={ADD_TEAM_ROLE_FIELD_ID}>Member Role</FieldLabel>
                <Select value={selectedRole} onValueChange={(value) => value !== null && setSelectedRole(value)}>
                  <SelectTrigger id={ADD_TEAM_ROLE_FIELD_ID} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MEMBER_ROLE_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value} title={option.value}>
                        <SimpleTooltip content={option.hint}>
                          <span className="font-medium">{option.value}</span>
                          <span className="ml-2 text-muted-foreground text-sm">- {option.hint}</span>
                        </SimpleTooltip>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            </FieldGroup>

            <div className="text-right mt-4">
              <Button type="submit" disabled={isAddingTeam || !selectedTeamId} aria-busy={isAddingTeam}>
                {isAddingTeam ? "Adding..." : "Add to Team"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
