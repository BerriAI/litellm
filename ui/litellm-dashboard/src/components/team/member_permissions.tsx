import {
  getTeamPermissionsCall,
  teamPermissionsUpdateCall,
} from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { RefreshCcw, Save } from "lucide-react";
import React, { useEffect, useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import { getPermissionInfo } from "./permission_definitions";

interface MemberPermissionsProps {
  teamId: string;
  accessToken: string | null;
  canEditTeam: boolean;
}

const MemberPermissions: React.FC<MemberPermissionsProps> = ({
  teamId,
  accessToken,
  canEditTeam,
}) => {
  const [permissions, setPermissions] = useState<string[]>([]);
  const [selectedPermissions, setSelectedPermissions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  const fetchPermissions = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await getTeamPermissionsCall(accessToken, teamId);
      const allPermissions = response.all_available_permissions || [];
      setPermissions(allPermissions);
      const teamPermissions = response.team_member_permissions || [];
      setSelectedPermissions(teamPermissions);
      setHasChanges(false);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to load permissions");
      console.error("Error fetching permissions:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPermissions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teamId, accessToken]);

  const handlePermissionChange = (permission: string, checked: boolean) => {
    const newSelectedPermissions = checked
      ? [...selectedPermissions, permission]
      : selectedPermissions.filter((p) => p !== permission);
    setSelectedPermissions(newSelectedPermissions);
    setHasChanges(true);
  };

  const handleSave = async () => {
    try {
      if (!accessToken) return;
      setSaving(true);
      await teamPermissionsUpdateCall(accessToken, teamId, selectedPermissions);
      NotificationsManager.success("Permissions updated successfully");
      setHasChanges(false);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update permissions");
      console.error("Error updating permissions:", error);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    fetchPermissions();
  };

  if (loading) {
    return <div className="p-6 text-center">Loading permissions...</div>;
  }

  const hasPermissions = permissions.length > 0;

  return (
    <Card className="bg-background shadow-md rounded-md p-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b border-border pb-4 mb-6">
        <h3 className="text-lg font-semibold mb-2 sm:mb-0">
          Member Permissions
        </h3>
        {canEditTeam && hasChanges && (
          <div className="flex gap-3">
            <Button variant="outline" onClick={handleReset}>
              <RefreshCcw className="h-4 w-4" />
              Reset
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              <Save className="h-4 w-4" />
              {saving ? "Saving…" : "Save Changes"}
            </Button>
          </div>
        )}
      </div>

      <p className="mb-6 text-muted-foreground">
        Control what team members can do when they are not team admins.
      </p>

      {hasPermissions ? (
        <div className="overflow-x-auto border border-border rounded-md">
          <Table className="min-w-full">
            <TableHeader>
              <TableRow>
                <TableHead>Method</TableHead>
                <TableHead>Endpoint</TableHead>
                <TableHead>Description</TableHead>
                <TableHead className="sticky right-0 bg-background shadow-[-4px_0_4px_-4px_rgba(0,0,0,0.1)] text-center">
                  Allow Access
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {permissions.map((permission) => {
                const permInfo = getPermissionInfo(permission);
                return (
                  <TableRow
                    key={permission}
                    className="hover:bg-muted transition-colors"
                  >
                    <TableCell>
                      <span
                        className={cn(
                          "px-2 py-1 rounded text-xs font-medium",
                          permInfo.method === "GET"
                            ? "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300"
                            : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
                        )}
                      >
                        {permInfo.method}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className="font-mono text-sm text-foreground">
                        {permInfo.endpoint}
                      </span>
                    </TableCell>
                    <TableCell className="text-foreground">
                      {permInfo.description}
                    </TableCell>
                    <TableCell className="sticky right-0 bg-background shadow-[-4px_0_4px_-4px_rgba(0,0,0,0.1)] text-center">
                      <Checkbox
                        checked={selectedPermissions.includes(permission)}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(permission, checked === true)
                        }
                        disabled={!canEditTeam}
                      />
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      ) : (
        <div className="py-12 text-center text-muted-foreground">
          No permissions available
        </div>
      )}
    </Card>
  );
};

export default MemberPermissions;
