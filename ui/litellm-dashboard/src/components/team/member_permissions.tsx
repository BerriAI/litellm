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
import { Button, message, Checkbox, Empty } from "antd";
import { getTeamPermissionsCall, teamPermissionsUpdateCall } from "@/components/networking";
import { getPermissionInfo, PermissionInfo } from "./permission_definitions";

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
      
      // Handle null or empty permissions
      const teamPermissions = response.team_member_permissions || [];
      setSelectedPermissions(teamPermissions);
      
      setHasChanges(false);
    } catch (error) {
      message.error("Failed to load permissions");
      console.error("Error fetching permissions:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPermissions();
  }, [teamId, accessToken]);

  const handlePermissionChange = (permission: string, checked: boolean) => {
    let newSelectedPermissions;
    
    if (checked) {
      newSelectedPermissions = [...selectedPermissions, permission];
    } else {
      newSelectedPermissions = selectedPermissions.filter(p => p !== permission);
    }
    
    setSelectedPermissions(newSelectedPermissions);
    setHasChanges(true);
  };

  const handleSave = async () => {
    try {
      if (!accessToken) return;
      
      setSaving(true);
      await teamPermissionsUpdateCall(accessToken, teamId, selectedPermissions);
      
      message.success("Permissions updated successfully");
      setHasChanges(false);
    } catch (error) {
      message.error("Failed to update permissions");
      console.error("Error updating permissions:", error);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    fetchPermissions();
  };

  if (loading) {
    return <div className="p-4">Loading permissions...</div>;
  }

  const hasPermissions = permissions.length > 0;

  return (
    <Card>
      <div className="flex justify-between items-center mb-4">
        <Title>Member Permissions</Title>
        {canEditTeam && hasChanges && (
          <div className="flex gap-2">
            <Button onClick={handleReset}>
              Reset
            </Button>
            <TremorButton
              onClick={handleSave}
              loading={saving}
            >
              Save changes
            </TremorButton>
          </div>
        )}
      </div>
      
      <Text className="mb-6">
        Control what team members can do when they are not admins.
      </Text>
      
      {hasPermissions ? (
        <Table className="mt-4">
          <TableHead>
            <TableRow>
              <TableHeaderCell>Method</TableHeaderCell>
              <TableHeaderCell>Endpoint</TableHeaderCell>
              <TableHeaderCell>Description</TableHeaderCell>
              <TableHeaderCell className="text-right">Access</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {permissions.map((permission) => {
              const permInfo = getPermissionInfo(permission);
              return (
                <TableRow key={permission}>
                  <TableCell>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      permInfo.method === 'GET' 
                        ? 'bg-blue-100 text-blue-800' 
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {permInfo.method}
                    </span>
                  </TableCell>
                  <TableCell>
                    <span className="font-mono text-sm">{permInfo.endpoint}</span>
                  </TableCell>
                  <TableCell>{permInfo.description}</TableCell>
                  <TableCell className="text-right">
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
      ) : (
        <div className="py-8">
          <Empty description="No permissions available" />
        </div>
      )}
      
      {canEditTeam && hasChanges && (
        <div className="flex justify-end mt-6">
          <TremorButton
            onClick={handleSave}
            loading={saving}
          >
            Save changes
          </TremorButton>
        </div>
      )}
    </Card>
  );
};

export default MemberPermissions;
