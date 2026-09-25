import React, { useId, useState } from "react";
import { userBulkUpdateUserCall, teamBulkMemberAddCall, Member } from "@/components/networking";
import { UserEditView } from "./user_edit_view";
import { toast } from "@/lib/toast";
import { MoneyCell } from "@/components/shared/table_cells";
import { MultiSelect } from "@/components/shared/MultiSelect";
import NumericalInput from "@/components/shared/numerical_input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface BulkEditUserModalProps {
  open: boolean;
  onCancel: () => void;
  selectedUsers: any[];
  possibleUIRoles: Record<string, Record<string, string>> | null;
  accessToken: string | null;
  onSuccess: () => void;
  teams: any[] | null;
  userRole: string | null;
  userModels: string[];
  allowAllUsers?: boolean; // Optional flag to enable "all users" mode
}

const BulkEditUserModal: React.FC<BulkEditUserModalProps> = ({
  open,
  onCancel,
  selectedUsers,
  possibleUIRoles,
  accessToken,
  onSuccess,
  teams,
  userRole,
  userModels,
  allowAllUsers = false,
}) => {
  const { premiumUser } = useAuthorized();
  const [loading, setLoading] = useState(false);
  const [selectedTeams, setSelectedTeams] = useState<string[]>([]);
  const [teamBudget, setTeamBudget] = useState<number | null>(null);
  const [addToTeams, setAddToTeams] = useState(false);
  const [updateAllUsers, setUpdateAllUsers] = useState(false);
  const updateAllUsersId = useId();
  const addToTeamsId = useId();
  const selectedTeamsId = useId();
  const teamBudgetId = useId();

  const handleCancel = () => {
    // Reset team management state
    setSelectedTeams([]);
    setTeamBudget(null);
    setAddToTeams(false);
    setUpdateAllUsers(false);
    onCancel();
  };

  // Create a mock userData object for the UserEditView
  const mockUserData = React.useMemo(
    () => ({
      user_id: "bulk_edit",
      user_info: {
        user_email: "",
        user_role: "",
        teams: [],
        models: [],
        max_budget: null,
        spend: 0,
        metadata: {},
        created_at: null,
        updated_at: null,
      },
      keys: [],
      teams: teams || [],
    }),
    [teams, open],
  );

  const handleSubmit = async (formValues: any) => {
    if (!accessToken) {
      toast.fromError("Access token not found");
      return;
    }

    setLoading(true);
    try {
      const userIds = selectedUsers.map((user) => user.user_id);

      // Build the update payload - only include fields that have been changed from default/empty values
      const updatePayload: any = {};

      if (formValues.user_role && formValues.user_role !== "") {
        updatePayload.user_role = formValues.user_role;
      }

      if (formValues.max_budget !== null && formValues.max_budget !== undefined) {
        updatePayload.max_budget = formValues.max_budget;
      }

      if (formValues.models && formValues.models.length > 0) {
        updatePayload.models = formValues.models;
      }

      if (formValues.budget_duration && formValues.budget_duration !== "") {
        updatePayload.budget_duration = formValues.budget_duration;
      }

      if (formValues.metadata && Object.keys(formValues.metadata).length > 0) {
        updatePayload.metadata = formValues.metadata;
      }

      // Check if any operations were requested
      const hasUserUpdates = Object.keys(updatePayload).length > 0;
      const hasTeamAdditions = addToTeams && selectedTeams.length > 0;

      if (!hasUserUpdates && !hasTeamAdditions) {
        toast.fromError("Please modify at least one field or select teams to add users to");
        return;
      }

      let successMessages: string[] = [];

      // Handle user property updates
      if (hasUserUpdates) {
        if (updateAllUsers) {
          const result = await userBulkUpdateUserCall(accessToken, updatePayload, undefined, true);
          successMessages.push(`Updated all users (${result.total_requested} total)`);
        } else {
          await userBulkUpdateUserCall(accessToken, updatePayload, userIds);
          successMessages.push(`Updated ${userIds.length} user(s)`);
        }
      }

      // Handle team additions
      if (hasTeamAdditions) {
        const teamResults: any[] = [];

        for (const teamId of selectedTeams) {
          try {
            // Create member objects for bulk add
            let members: Member[] | null = null;
            if (updateAllUsers) {
              members = null;
            } else {
              members = selectedUsers.map((user) => ({
                user_id: user.user_id,
                role: "user" as const, // Default role for bulk add
                user_email: user.user_email || null,
              }));
            }

            const result = await teamBulkMemberAddCall(
              accessToken,
              teamId,
              members ? members : null,
              teamBudget || undefined,
              updateAllUsers,
            );

            teamResults.push({
              teamId,
              success: true,
              successfulAdditions: result.successful_additions,
              failedAdditions: result.failed_additions,
            });
          } catch (error) {
            console.error(`Failed to add users to team ${teamId}:`, error);
            teamResults.push({
              teamId,
              success: false,
              error: error,
            });
          }
        }

        // Generate team success message
        const successfulTeams = teamResults.filter((r) => r.success);
        const failedTeams = teamResults.filter((r) => !r.success);

        if (successfulTeams.length > 0) {
          const totalAdditions = successfulTeams.reduce((sum, r) => sum + r.successfulAdditions, 0);
          successMessages.push(`Added users to ${successfulTeams.length} team(s) (${totalAdditions} total additions)`);
        }

        if (failedTeams.length > 0) {
          toast.warning(`Failed to add users to ${failedTeams.length} team(s)`);
        }
      }

      if (successMessages.length > 0) {
        toast.success(successMessages.join(". "));
      }

      // Reset team management state
      setSelectedTeams([]);
      setTeamBudget(null);
      setAddToTeams(false);
      setUpdateAllUsers(false);

      onSuccess();
      onCancel();
    } catch (error) {
      console.error("Bulk operation failed:", error);
      toast.fromError("Failed to perform bulk operations");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>
            {updateAllUsers ? "Bulk Edit All Users" : `Bulk Edit ${selectedUsers.length} User(s)`}
          </DialogTitle>
        </DialogHeader>
        {allowAllUsers && (
          <div className="mb-4">
            <div className="flex items-center gap-2">
              <Checkbox
                id={updateAllUsersId}
                checked={updateAllUsers}
                onCheckedChange={(checked) => setUpdateAllUsers(checked === true)}
                aria-label="Update ALL users in the system"
              />
              <label htmlFor={updateAllUsersId} className="cursor-pointer text-sm font-medium text-foreground">
                Update ALL users in the system
              </label>
            </div>
            {updateAllUsers && (
              <div className="mt-2">
                <span className="text-xs text-warning">
                  ⚠️ This will apply changes to ALL users in the system, not just the selected ones.
                </span>
              </div>
            )}
          </div>
        )}

        {!updateAllUsers && (
          <div className="mb-4">
            <h5 className="mb-2 text-sm font-semibold text-foreground">Selected Users ({selectedUsers.length}):</h5>
            <div className="max-h-[200px] overflow-y-auto rounded-md border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[30%]">User ID</TableHead>
                    <TableHead className="w-[25%]">Email</TableHead>
                    <TableHead className="w-[25%]">Current Role</TableHead>
                    <TableHead className="w-[20%]">Budget</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {selectedUsers.map((user) => (
                    <TableRow key={user.user_id}>
                      <TableCell className="text-xs font-medium text-foreground">
                        {user.user_id.length > 20 ? `${user.user_id.slice(0, 20)}...` : user.user_id}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">{user.user_email || "No email"}</TableCell>
                      <TableCell className="text-xs text-foreground">
                        {possibleUIRoles?.[user.user_role]?.ui_label || user.user_role}
                      </TableCell>
                      <TableCell>
                        <MoneyCell value={user.max_budget} decimals={2} emptyText="Unlimited" showZero />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}

        <Separator className="my-6" />

        <div className="mb-4">
          <p className="text-sm text-foreground">
            <strong>Instructions:</strong> Fill in the fields below with the values you want to apply to all selected
            users. You can bulk edit: role, budget, models, and metadata. You can also add users to teams.
          </p>
        </div>

        {/* Team Management Section */}
        <Card size="sm" className="mb-4 bg-muted/50">
          <CardHeader>
            <CardTitle>Team Management</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <Checkbox
                  id={addToTeamsId}
                  checked={addToTeams}
                  onCheckedChange={(checked) => setAddToTeams(checked === true)}
                  aria-label="Add selected users to teams"
                />
                <label htmlFor={addToTeamsId} className="cursor-pointer text-sm text-foreground">
                  Add selected users to teams
                </label>
              </div>

              {addToTeams && (
                <>
                  <div>
                    <label htmlFor={selectedTeamsId} className="block text-sm font-medium text-foreground">
                      Select Teams:
                    </label>
                    <MultiSelect
                      id={selectedTeamsId}
                      className="mt-2"
                      placeholder="Select teams to add users to"
                      value={selectedTeams}
                      onValueChange={setSelectedTeams}
                      options={
                        teams?.map((team) => ({
                          label: team.team_alias || team.team_id,
                          value: team.team_id,
                        })) || []
                      }
                    />
                  </div>

                  <div>
                    <label htmlFor={teamBudgetId} className="block text-sm font-medium text-foreground">
                      Team Budget (Optional):
                    </label>
                    <NumericalInput
                      id={teamBudgetId}
                      className="mt-2"
                      placeholder="Max budget per user in team"
                      value={teamBudget ?? ""}
                      onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                        setTeamBudget(event.target.value === "" ? null : Number(event.target.value))
                      }
                      min={0}
                      step={0.01}
                    />
                    <span className="text-xs text-muted-foreground">
                      Leave empty for unlimited budget within team limits
                    </span>
                  </div>

                  <span className="text-xs text-muted-foreground">
                    Users will be added with &quot;user&quot; role by default. All users will be added to each selected
                    team.
                  </span>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        <UserEditView
          userData={mockUserData}
          onCancel={handleCancel}
          onSubmit={handleSubmit}
          teams={teams}
          accessToken={accessToken}
          userID="bulk_edit"
          userRole={userRole}
          userModels={userModels}
          possibleUIRoles={possibleUIRoles}
          isBulkEdit={true}
          premiumUser={premiumUser === true}
        />

        {loading && (
          <div className="mt-2.5 text-center">
            <span className="text-sm text-foreground">
              Updating {updateAllUsers ? "all users" : selectedUsers.length} user(s)...
            </span>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default BulkEditUserModal;
