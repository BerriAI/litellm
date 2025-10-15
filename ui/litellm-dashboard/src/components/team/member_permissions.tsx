import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Button as TremorButton,
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
} from "@tremor/react";
import { Button, Checkbox, Empty } from "antd";
import { ReloadOutlined, SaveOutlined } from "@ant-design/icons";
import { getTeamPermissionsCall, teamPermissionsUpdateCall } from "@/components/networking";
import { getPermissionInfo } from "./permission_definitions";
import NotificationsManager from "../molecules/notifications_manager";

interface MemberPermissionsProps {
  teamId: string;
  accessToken: string | null;
  canEditTeam: boolean;
}

const MemberPermissions: React.FC<MemberPermissionsProps> = ({ teamId, accessToken, canEditTeam }) => {
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
    <Card className="bg-white shadow-md rounded-md p-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b pb-4 mb-6">
        <Title className="mb-2 sm:mb-0">Member Permissions</Title>
        {canEditTeam && hasChanges && (
          <div className="flex gap-3">
            <Button icon={<ReloadOutlined />} onClick={handleReset}>
              Reset
            </Button>
            <TremorButton onClick={handleSave} loading={saving} className="flex items-center gap-2">
              <SaveOutlined /> Save Changes
            </TremorButton>
          </div>
        )}
      </div>

      <Text className="mb-6 text-gray-600">Control what team members can do when they are not team admins.</Text>

      {hasPermissions ? (
        <div className="overflow-x-auto">
          <Table className=" min-w-full">
            <TableHead>
              <TableRow>
                <TableHeaderCell>Method</TableHeaderCell>
                <TableHeaderCell>Endpoint</TableHeaderCell>
                <TableHeaderCell>Description</TableHeaderCell>
                <TableHeaderCell className="sticky right-0 bg-white shadow-[-4px_0_4px_-4px_rgba(0,0,0,0.1)] text-center">
                  Allow Access
                </TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {permissions.map((permission) => {
                const permInfo = getPermissionInfo(permission);
                return (
                  <TableRow key={permission} className="hover:bg-gray-50 transition-colors">
                    <TableCell>
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          permInfo.method === "GET" ? "bg-blue-100 text-blue-800" : "bg-green-100 text-green-800"
                        }`}
                      >
                        {permInfo.method}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className="font-mono text-sm text-gray-800">{permInfo.endpoint}</span>
                    </TableCell>
                    <TableCell className="text-gray-700">{permInfo.description}</TableCell>
                    <TableCell className="sticky right-0 bg-white shadow-[-4px_0_4px_-4px_rgba(0,0,0,0.1)] text-center">
                      <Checkbox
                        checked={selectedPermissions.includes(permission)}
                        onChange={(e) => handlePermissionChange(permission, e.target.checked)}
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
          <Empty description="No permissions available" />
        </div>
      )}
    </Card>
  );
};

export default MemberPermissions;
