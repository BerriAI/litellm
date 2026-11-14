import { getTeamPermissionsCall, teamPermissionsUpdateCall } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { RotateCw, Save } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "@/lib/toast";
import { getPermissionInfo } from "./permission_definitions";

interface MemberPermissionsProps {
  teamId: string;
  accessToken: string | null;
  canEditTeam: boolean;
}

const MemberPermissions: React.FC<MemberPermissionsProps> = ({ teamId, accessToken, canEditTeam }) => {
  const { t } = useTranslation();
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
      toast.fromError(t("teams.memberPermissions.toasts.loadFailed"));
      console.error("Error fetching permissions:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPermissions();
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
      toast.success(t("teams.memberPermissions.toasts.updated"));
      setHasChanges(false);
    } catch (error) {
      toast.fromError(t("teams.memberPermissions.toasts.updateFailed"));
      console.error("Error updating permissions:", error);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    fetchPermissions();
  };

  if (loading) {
    return <div className="p-6 text-center">{t("teams.memberPermissions.loading")}</div>;
  }

  const hasPermissions = permissions.length > 0;

  return (
    <Card className="block bg-card shadow-md rounded-md p-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b pb-4 mb-6">
        <h3 className="text-lg font-medium text-foreground mb-2 sm:mb-0">{t("teams.memberPermissions.title")}</h3>
        {canEditTeam && hasChanges && (
          <div className="flex gap-3">
            <Button variant="outline" onClick={handleReset}>
              <RotateCw className="size-3.5" />
              {t("teams.memberPermissions.reset")}
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              <Save className="size-3.5" />
              {t("teams.memberPermissions.save")}
            </Button>
          </div>
        )}
      </div>

      <p className="mb-6 text-sm text-muted-foreground">{t("teams.memberPermissions.description")}</p>

      {hasPermissions ? (
        <div className="overflow-x-auto">
          <Table className="min-w-full">
            <TableHeader>
              <TableRow>
                <TableHead>{t("teams.memberPermissions.columns.method")}</TableHead>
                <TableHead>{t("teams.memberPermissions.columns.endpoint")}</TableHead>
                <TableHead>{t("teams.memberPermissions.columns.description")}</TableHead>
                <TableHead className="sticky right-0 bg-card shadow-[-4px_0_4px_-4px_rgba(0,0,0,0.1)] text-center">
                  {t("teams.memberPermissions.columns.allowAccess")}
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {permissions.map((permission) => {
                const permInfo = getPermissionInfo(permission);
                return (
                  <TableRow key={permission} className="hover:bg-accent transition-colors">
                    <TableCell>
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          permInfo.method === "GET" ? "bg-info/15 text-info" : "bg-success/15 text-success"
                        }`}
                      >
                        {permInfo.method}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className="font-mono text-sm text-foreground">{permInfo.endpoint}</span>
                    </TableCell>
                    <TableCell className="text-foreground">{permInfo.description}</TableCell>
                    <TableCell className="sticky right-0 bg-card shadow-[-4px_0_4px_-4px_rgba(0,0,0,0.1)] text-center">
                      <Checkbox
                        className="mx-auto"
                        checked={selectedPermissions.includes(permission)}
                        onCheckedChange={(checked) => handlePermissionChange(permission, checked)}
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
        <div className="py-12">
          <p className="text-center text-sm text-muted-foreground">{t("teams.memberPermissions.empty")}</p>
        </div>
      )}
    </Card>
  );
};

export default MemberPermissions;
