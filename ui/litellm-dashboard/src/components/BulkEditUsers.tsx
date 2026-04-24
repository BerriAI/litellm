import React, { useMemo, useState } from "react";
import { X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import MessageManager from "@/components/molecules/message_manager";
import {
  userBulkUpdateUserCall,
  teamBulkMemberAddCall,
  Member,
} from "./networking";
import { UserEditView } from "./user_edit_view";
import NotificationsManager from "./molecules/notifications_manager";
import NumericalInput from "./shared/numerical_input";

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

/**
 * shadcn Select + Badge chip multi-select for teams.
 */
function TeamMultiSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No more teams available
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
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
  const [loading, setLoading] = useState(false);
  const [selectedTeams, setSelectedTeams] = useState<string[]>([]);
  const [teamBudget, setTeamBudget] = useState<number | null>(null);
  const [addToTeams, setAddToTeams] = useState(false);
  const [updateAllUsers, setUpdateAllUsers] = useState(false);

  const handleCancel = () => {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [teams, open],
  );

  const handleSubmit = async (formValues: any) => {
    console.log("formValues", formValues);
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token not found");
      return;
    }

    setLoading(true);
    try {
      const userIds = selectedUsers.map((user) => user.user_id);

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

      if (
        formValues.metadata &&
        Object.keys(formValues.metadata).length > 0
      ) {
        updatePayload.metadata = formValues.metadata;
      }

      const hasUserUpdates = Object.keys(updatePayload).length > 0;
      const hasTeamAdditions = addToTeams && selectedTeams.length > 0;

      if (!hasUserUpdates && !hasTeamAdditions) {
        NotificationsManager.fromBackend(
          "Please modify at least one field or select teams to add users to",
        );
        return;
      }

      let successMessages: string[] = [];

      if (hasUserUpdates) {
        if (updateAllUsers) {
          const result = await userBulkUpdateUserCall(
            accessToken,
            updatePayload,
            undefined,
            true,
          );
          successMessages.push(
            `Updated all users (${result.total_requested} total)`,
          );
        } else {
          await userBulkUpdateUserCall(accessToken, updatePayload, userIds);
          successMessages.push(`Updated ${userIds.length} user(s)`);
        }
      }

      if (hasTeamAdditions) {
        const teamResults: any[] = [];

        for (const teamId of selectedTeams) {
          try {
            let members: Member[] | null = null;
            if (updateAllUsers) {
              members = null;
            } else {
              members = selectedUsers.map((user) => ({
                user_id: user.user_id,
                role: "user" as const,
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

            console.log("result", result);

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

        const successfulTeams = teamResults.filter((r) => r.success);
        const failedTeams = teamResults.filter((r) => !r.success);

        if (successfulTeams.length > 0) {
          const totalAdditions = successfulTeams.reduce(
            (sum, r) => sum + r.successfulAdditions,
            0,
          );
          successMessages.push(
            `Added users to ${successfulTeams.length} team(s) (${totalAdditions} total additions)`,
          );
        }

        if (failedTeams.length > 0) {
          MessageManager.warning(
            `Failed to add users to ${failedTeams.length} team(s)`,
          );
        }
      }

      if (successMessages.length > 0) {
        NotificationsManager.success(successMessages.join(". "));
      }

      setSelectedTeams([]);
      setTeamBudget(null);
      setAddToTeams(false);
      setUpdateAllUsers(false);

      onSuccess();
      onCancel();
    } catch (error) {
      console.error("Bulk operation failed:", error);
      NotificationsManager.fromBackend("Failed to perform bulk operations");
    } finally {
      setLoading(false);
    }
  };

  const teamOptions = (teams || []).map((team: any) => ({
    label: team.team_alias || team.team_id,
    value: team.team_id,
  }));

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-[800px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {updateAllUsers
              ? "Bulk Edit All Users"
              : `Bulk Edit ${selectedUsers.length} User(s)`}
          </DialogTitle>
        </DialogHeader>

        {allowAllUsers && (
          <div className="mb-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={updateAllUsers}
                onCheckedChange={(c) => setUpdateAllUsers(c === true)}
              />
              <span className="font-semibold">
                Update ALL users in the system
              </span>
            </label>
            {updateAllUsers && (
              <div className="mt-2">
                <span className="text-xs text-amber-600 dark:text-amber-400">
                  ⚠️ This will apply changes to ALL users in the system, not
                  just the selected ones.
                </span>
              </div>
            )}
          </div>
        )}

        {!updateAllUsers && (
          <div className="mb-4">
            <h5 className="text-sm font-semibold mb-2">
              Selected Users ({selectedUsers.length}):
            </h5>
            <div className="border border-border rounded-md max-h-[200px] overflow-y-auto">
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
                      <TableCell className="text-xs font-semibold">
                        {user.user_id.length > 20
                          ? `${user.user_id.slice(0, 20)}...`
                          : user.user_id}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {user.user_email || "No email"}
                      </TableCell>
                      <TableCell className="text-xs">
                        {possibleUIRoles?.[user.user_role]?.ui_label ||
                          user.user_role}
                      </TableCell>
                      <TableCell className="text-xs">
                        {user.max_budget !== null
                          ? `$${user.max_budget}`
                          : "Unlimited"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}

        <Separator />

        <div className="mb-4">
          <p className="text-sm">
            <strong>Instructions:</strong> Fill in the fields below with the
            values you want to apply to all selected users. You can bulk edit:
            role, budget, models, and metadata. You can also add users to
            teams.
          </p>
        </div>

        {/* Team Management Section */}
        <Card className="p-4 bg-muted/50 mb-4">
          <h4 className="text-sm font-semibold mb-3">Team Management</h4>
          <div className="flex flex-col gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={addToTeams}
                onCheckedChange={(c) => setAddToTeams(c === true)}
              />
              <span>Add selected users to teams</span>
            </label>

            {addToTeams && (
              <>
                <div>
                  <Label className="font-semibold">Select Teams:</Label>
                  <div className="mt-2">
                    <TeamMultiSelect
                      value={selectedTeams}
                      onChange={setSelectedTeams}
                      options={teamOptions}
                      placeholder="Select teams to add users to"
                    />
                  </div>
                </div>

                <div>
                  <Label className="font-semibold">Team Budget (Optional):</Label>
                  <div className="mt-2">
                    <NumericalInput
                      placeholder="Max budget per user in team"
                      value={teamBudget}
                      onChange={(v: number | null) => setTeamBudget(v)}
                      min={0}
                      step={0.01}
                      precision={2}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground">
                    Leave empty for unlimited budget within team limits
                  </span>
                </div>

                <span className="text-xs text-muted-foreground">
                  Users will be added with &quot;user&quot; role by default.
                  All users will be added to each selected team.
                </span>
              </>
            )}
          </div>
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
        />

        {loading && (
          <div className="text-center mt-3">
            <span className="text-sm">
              Updating{" "}
              {updateAllUsers ? "all users" : selectedUsers.length} user(s)...
            </span>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default BulkEditUserModal;
