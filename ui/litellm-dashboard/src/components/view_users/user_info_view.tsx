import React, { useState } from "react";
import {
  ArrowLeft as ArrowLeftIcon,
  Trash2 as TrashIcon,
  RefreshCcw as RefreshIcon,
  Plus as PlusIcon,
  Copy as CopyIcon,
  Check as CheckIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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
} from "../networking";
import { rolesWithWriteAccess } from "../../utils/roles";
import { UserEditView } from "../user_edit_view";
import OnboardingModal, { InvitationLink } from "../onboarding_link";
import {
  formatNumberWithCommas,
  copyToClipboard as utilCopyToClipboard,
} from "@/utils/dataUtils";
import NotificationsManager from "../molecules/notifications_manager";
import { getBudgetDurationLabel } from "../common_components/budget_duration_dropdown";
import DeleteResourceModal from "../common_components/DeleteResourceModal";

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
  const [userData, setUserData] = useState<UserInfoV2Response | null>(null);
  const [teamDetails, setTeamDetails] = useState<TeamDisplayInfo[]>([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeletingUser, setIsDeletingUser] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(startInEditMode);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] =
    useState(false);
  const [invitationLinkData, setInvitationLinkData] =
    useState<InvitationLink | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(
    initialTab === 1 ? "details" : "overview",
  );
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [isTeamsExpanded, setIsTeamsExpanded] = useState(false);
  const [isAddTeamModalOpen, setIsAddTeamModalOpen] = useState(false);
  const [isRemoveTeamModalOpen, setIsRemoveTeamModalOpen] = useState(false);
  const [teamToRemove, setTeamToRemove] = useState<TeamDisplayInfo | null>(
    null,
  );
  const [isAddingTeam, setIsAddingTeam] = useState(false);
  const [isRemovingTeam, setIsRemovingTeam] = useState(false);
  const [allTeams, setAllTeams] = useState<
    Array<{ team_id: string; team_alias: string }>
  >([]);
  const [selectedTeamId, setSelectedTeamId] = useState<string>("");
  const [selectedRole, setSelectedRole] = useState<string>("user");
  const [isLoadingTeams, setIsLoadingTeams] = useState(false);

  React.useEffect(() => {
    setBaseUrl(getProxyBaseUrl());
  }, []);

  React.useEffect(() => {
    console.log(
      `userId: ${userId}, userRole: ${userRole}, accessToken: ${accessToken}`,
    );
    const fetchData = async () => {
      try {
        if (!accessToken) return;
        const data = await userGetInfoV2(accessToken, userId);
        setUserData(data);

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
            setTeamDetails(
              data.teams.map((id: string) => ({
                team_id: id,
                team_alias: null,
              })),
            );
          }
        }

        const modelDataResponse = await modelAvailableCall(
          accessToken,
          userId,
          userRole || "",
        );
        const availableModels = modelDataResponse.data.map(
          (model: any) => model.id,
        );
        setUserModels(availableModels);
      } catch (error) {
        console.error("Error fetching user data:", error);
        NotificationsManager.fromBackend("Failed to fetch user data");
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

  const refreshTeams = async () => {
    if (!accessToken) return;
    const data = await userGetInfoV2(accessToken, userId);
    setUserData(data);
    if (data.teams && data.teams.length > 0) {
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
      setTeamDetails(await Promise.all(teamPromises));
    } else {
      setTeamDetails([]);
    }
  };

  const handleAddTeamSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!accessToken || !selectedTeamId) return;
    setIsAddingTeam(true);
    try {
      const member: Member = {
        role: selectedRole,
        user_id: userId,
      };
      await teamMemberAddCall(accessToken, selectedTeamId, member);
      NotificationsManager.success("User added to team successfully");
      setIsAddTeamModalOpen(false);
      await refreshTeams();
    } catch (error: any) {
      console.error("Error adding user to team:", error);
      NotificationsManager.fromBackend(
        error?.message || "Failed to add user to team",
      );
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
      NotificationsManager.success("User removed from team successfully");
      setIsRemoveTeamModalOpen(false);
      setTeamToRemove(null);
      await refreshTeams();
    } catch (error: any) {
      console.error("Error removing user from team:", error);
      NotificationsManager.fromBackend(
        error?.message || "Failed to remove user from team",
      );
    } finally {
      setIsRemovingTeam(false);
    }
  };

  const handleRemoveTeamCancel = () => {
    setIsRemoveTeamModalOpen(false);
    setTeamToRemove(null);
  };

  const availableTeamsForAdd = allTeams.filter(
    (t) => !teamDetails.some((td) => td.team_id === t.team_id),
  );

  const handleResetPassword = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token not found");
      return;
    }
    try {
      NotificationsManager.success("Generating password reset link...");
      const data = await invitationCreateCall(accessToken, userId);
      setInvitationLinkData(data);
      setIsInvitationLinkModalVisible(true);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to generate password reset link");
    }
  };

  const handleDelete = async () => {
    try {
      if (!accessToken) return;
      setIsDeletingUser(true);
      await userDeleteCall(accessToken, [userId]);
      NotificationsManager.success("User deleted successfully");
      if (onDelete) {
        onDelete();
      }
      onClose();
    } catch (error) {
      console.error("Error deleting user:", error);
      NotificationsManager.fromBackend("Failed to delete user");
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

      // Response is unused here; local state is updated from form values
      await userUpdateUserCall(accessToken, formValues, null);

      setUserData({
        ...userData,
        user_email: formValues.user_email ?? userData.user_email,
        user_alias: formValues.user_alias ?? userData.user_alias,
        models: formValues.models ?? userData.models,
        max_budget: formValues.max_budget ?? userData.max_budget,
        budget_duration: formValues.budget_duration ?? userData.budget_duration,
        metadata: formValues.metadata ?? userData.metadata,
      });

      NotificationsManager.success("User updated successfully");
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating user:", error);
      NotificationsManager.fromBackend("Failed to update user");
    }
  };

  if (isLoading) {
    return (
      <div className="p-4">
        <Button
          variant="ghost"
          onClick={onClose}
          className="mb-4 flex items-center gap-1"
        >
          <ArrowLeftIcon className="h-4 w-4" />
          Back to Users
        </Button>
        <span>Loading user data...</span>
      </div>
    );
  }

  if (!userData) {
    return (
      <div className="p-4">
        <Button
          variant="ghost"
          onClick={onClose}
          className="mb-4 flex items-center gap-1"
        >
          <ArrowLeftIcon className="h-4 w-4" />
          Back to Users
        </Button>
        <span>User not found</span>
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
    },
  };

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button
            variant="ghost"
            onClick={onClose}
            className="mb-4 flex items-center gap-1"
          >
            <ArrowLeftIcon className="h-4 w-4" />
            Back to Users
          </Button>
          <h2 className="text-2xl font-semibold m-0">
            {userData.user_email || "User"}
          </h2>
          <div className="flex items-center cursor-pointer">
            <span className="text-muted-foreground font-mono">
              {userData.user_id}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(userData.user_id, "user-id")}
              className={`ml-2 z-10 h-7 w-7 p-0 transition-all duration-200 ${
                copiedStates["user-id"]
                  ? "text-green-600 bg-green-50 border-green-200 dark:bg-green-950/30"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
              aria-label="Copy user id"
            >
              {copiedStates["user-id"] ? (
                <CheckIcon size={12} />
              ) : (
                <CopyIcon size={12} />
              )}
            </Button>
          </div>
        </div>
        {userRole && rolesWithWriteAccess.includes(userRole) && (
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              onClick={handleResetPassword}
              className="flex items-center"
            >
              <RefreshIcon className="h-4 w-4 mr-1" />
              Reset Password
            </Button>
            <Button
              variant="outline"
              onClick={() => setIsDeleteModalOpen(true)}
              className="flex items-center text-destructive border-destructive hover:text-destructive hover:border-destructive"
            >
              <TrashIcon className="h-4 w-4 mr-1" />
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
            value:
              (userData.user_role &&
                possibleUIRoles?.[userData.user_role]?.ui_label) ||
              userData.user_role ||
              "-",
          },
          {
            label: "Total Spend (USD)",
            value:
              userData.spend !== null && userData.spend !== undefined
                ? userData.spend.toFixed(2)
                : undefined,
          },
        ]}
        onCancel={cancelDelete}
        onOk={handleDelete}
        confirmLoading={isDeletingUser}
      />

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="p-4">
              <span>Spend</span>
              <div className="mt-2">
                <h3 className="text-xl font-semibold m-0">
                  ${formatNumberWithCommas(userData.spend || 0, 4)}
                </h3>
                <span>
                  of{" "}
                  {userData.max_budget !== null
                    ? `$${formatNumberWithCommas(userData.max_budget, 4)}`
                    : "Unlimited"}
                </span>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex justify-between items-center mb-2">
                <span>Teams</span>
                {isProxyAdmin && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleOpenAddTeamModal}
                    className="h-7 gap-1"
                  >
                    <PlusIcon className="h-4 w-4" />
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
                          {isProxyAdmin && (
                            <TableHead className="text-right">
                              Actions
                            </TableHead>
                          )}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {teamDetails
                          .slice(
                            0,
                            isTeamsExpanded ? teamDetails.length : 20,
                          )
                          .map((team) => (
                            <TableRow key={team.team_id}>
                              <TableCell>
                                {team.team_alias || team.team_id}
                              </TableCell>
                              {isProxyAdmin && (
                                <TableCell className="text-right">
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() =>
                                      handleOpenRemoveTeamModal(team)
                                    }
                                    aria-label={`Remove ${team.team_alias || team.team_id}`}
                                    className="h-7 w-7 p-0 text-destructive hover:text-destructive"
                                  >
                                    <TrashIcon className="h-4 w-4" />
                                  </Button>
                                </TableCell>
                              )}
                            </TableRow>
                          ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <span>No teams</span>
                )}
                {!isTeamsExpanded && teamDetails.length > 20 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="mt-2"
                    onClick={() => setIsTeamsExpanded(true)}
                  >
                    +{teamDetails.length - 20} more
                  </Button>
                )}
                {isTeamsExpanded && teamDetails.length > 20 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="mt-2"
                    onClick={() => setIsTeamsExpanded(false)}
                  >
                    Show Less
                  </Button>
                )}
              </div>
            </Card>

            <Card className="p-4">
              <span>Personal Models</span>
              <div className="mt-2">
                {userData.models?.length && userData.models?.length > 0 ? (
                  userData.models?.map((model, index) => (
                    <div key={index}>{model}</div>
                  ))
                ) : (
                  <span>All proxy models</span>
                )}
              </div>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="details">
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold m-0">User Settings</h3>
              {!isEditing &&
                userRole &&
                rolesWithWriteAccess.includes(userRole) && (
                  <Button onClick={() => setIsEditing(true)}>
                    Edit Settings
                  </Button>
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
              />
            ) : (
              <div className="space-y-4">
                <div>
                  <span className="font-medium block">User ID</span>
                  <div className="flex items-center cursor-pointer">
                    <span className="font-mono">{userData.user_id}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        copyToClipboard(userData.user_id, "user-id-details")
                      }
                      className={`ml-2 z-10 h-7 w-7 p-0 transition-all duration-200 ${
                        copiedStates["user-id-details"]
                          ? "text-green-600 bg-green-50 border-green-200 dark:bg-green-950/30"
                          : "text-muted-foreground hover:text-foreground hover:bg-muted"
                      }`}
                      aria-label="Copy user id"
                    >
                      {copiedStates["user-id-details"] ? (
                        <CheckIcon size={12} />
                      ) : (
                        <CopyIcon size={12} />
                      )}
                    </Button>
                  </div>
                </div>

                <div>
                  <span className="font-medium block">Email</span>
                  <span>{userData.user_email || "Not Set"}</span>
                </div>

                <div>
                  <span className="font-medium block">User Alias</span>
                  <span>{userData.user_alias || "Not Set"}</span>
                </div>

                <div>
                  <span className="font-medium block">Global Proxy Role</span>
                  <span>{userData.user_role || "Not Set"}</span>
                </div>

                <div>
                  <span className="font-medium block">Created</span>
                  <span>
                    {userData.created_at
                      ? new Date(userData.created_at).toLocaleString()
                      : "Unknown"}
                  </span>
                </div>

                <div>
                  <span className="font-medium block">Last Updated</span>
                  <span>
                    {userData.updated_at
                      ? new Date(userData.updated_at).toLocaleString()
                      : "Unknown"}
                  </span>
                </div>

                <div>
                  <span className="font-medium block">Personal Models</span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {userData.models?.length && userData.models?.length > 0 ? (
                      userData.models?.map((model, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-blue-100 dark:bg-blue-950/50 text-blue-800 dark:text-blue-300 rounded text-xs"
                        >
                          {model}
                        </span>
                      ))
                    ) : (
                      <span>All proxy models</span>
                    )}
                  </div>
                </div>

                <div>
                  <span className="font-medium block">Max Budget</span>
                  <span>
                    {userData.max_budget !== null &&
                    userData.max_budget !== undefined
                      ? `$${formatNumberWithCommas(userData.max_budget, 4)}`
                      : "Unlimited"}
                  </span>
                </div>

                <div>
                  <span className="font-medium block">Budget Reset</span>
                  <span>
                    {getBudgetDurationLabel(userData.budget_duration ?? null)}
                  </span>
                </div>

                <div>
                  <span className="font-medium block">Metadata</span>
                  <pre className="bg-muted p-2 rounded text-xs overflow-auto mt-1">
                    {JSON.stringify(userData.metadata || {}, null, 2)}
                  </pre>
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
          {
            label: "Team",
            value: teamToRemove?.team_alias || teamToRemove?.team_id,
          },
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
        onOpenChange={(o) =>
          !isAddingTeam && !o ? setIsAddTeamModalOpen(false) : undefined
        }
      >
        <DialogContent className="max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Add User to Team</DialogTitle>
            <DialogDescription className="sr-only">
              Add the user to an existing team.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleAddTeamSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="add-team-team">
                Team <span className="text-destructive">*</span>
              </Label>
              <Select
                value={selectedTeamId}
                onValueChange={setSelectedTeamId}
                disabled={isLoadingTeams}
              >
                <SelectTrigger id="add-team-team">
                  <SelectValue placeholder="Select a team" />
                </SelectTrigger>
                <SelectContent>
                  {availableTeamsForAdd.map((team) => (
                    <SelectItem key={team.team_id} value={team.team_id}>
                      {team.team_alias}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="add-team-role">Member Role</Label>
              <Select value={selectedRole} onValueChange={setSelectedRole}>
                <SelectTrigger id="add-team-role">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span>
                            <span className="font-medium">user</span>
                            <span className="ml-2 text-muted-foreground text-sm">
                              - Can view team info, but not manage it
                            </span>
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>
                          Can view team info, but not manage it
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </SelectItem>
                  <SelectItem value="admin">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span>
                            <span className="font-medium">admin</span>
                            <span className="ml-2 text-muted-foreground text-sm">
                              - Can create team keys, add members, and manage
                              settings
                            </span>
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>
                          Can create team keys, add members, and manage
                          settings
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <DialogFooter>
              <Button
                type="submit"
                disabled={!selectedTeamId || isAddingTeam}
              >
                {isAddingTeam ? "Adding..." : "Add to Team"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
