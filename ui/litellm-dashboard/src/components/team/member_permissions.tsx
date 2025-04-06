import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Button as TremorButton,
  Divider,
} from "@tremor/react";
import { Button, message, Checkbox, List, Empty } from "antd";
import { getTeamPermissionsCall, teamPermissionsUpdateCall } from "@/components/networking";
import { CheckOutlined, KeyOutlined, UserOutlined, InfoCircleOutlined } from "@ant-design/icons";

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
      message.error("Failed to load permissions");
      console.error("Error fetching permissions:", error);
    } finally {
      setLoading(false);
    }
  };

  const getGroupedPermissions = (): Record<string, string[]> => {
    const groups: Record<string, string[]> = {};
    
    permissions.forEach(permission => {
      const parts = permission.split('/');
      if (parts.length > 1) {
        const groupName = `/${parts[1]}`;
        if (!groups[groupName]) {
          groups[groupName] = [];
        }
        groups[groupName].push(permission);
      } else {
        if (!groups['Other']) {
          groups['Other'] = [];
        }
        groups['Other'].push(permission);
      }
    });
    
    return groups;
  };

  const getGroupIcon = (groupName: string) => {
    if (groupName === '/key') return <KeyOutlined />;
    if (groupName === '/user') return <UserOutlined />;
    return <InfoCircleOutlined />;
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

  const groupedPermissions = getGroupedPermissions();
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
      
      <Text className="mb-4">
        Control what team members can do when they are not admins.
      </Text>
      
      {hasPermissions ? (
        <>
          <Divider />
          
          <div className="mt-6">
            {Object.entries(groupedPermissions).map(([groupName, groupPermissions]) => (
              <div key={groupName} className="mb-6">
                <div className="flex items-center mb-2">
                  <span className="mr-2">{getGroupIcon(groupName)}</span>
                  <Text className="font-semibold">{groupName} Permissions</Text>
                </div>
                
                <List
                  itemLayout="horizontal"
                  dataSource={groupPermissions}
                  renderItem={(permission: string) => (
                    <List.Item className="py-3 hover:bg-gray-50">
                      <div className="flex items-center w-full">
                        <div className="mr-3">
                          {selectedPermissions.includes(permission) && 
                            <CheckOutlined style={{ color: '#1677ff' }} />
                          }
                        </div>
                        <div className="flex-grow">
                          <Checkbox
                            checked={selectedPermissions.includes(permission)}
                            onChange={(e) => handlePermissionChange(permission, e.target.checked)}
                            disabled={!canEditTeam}
                          >
                            <Text className="font-mono">{permission}</Text>
                          </Checkbox>
                        </div>
                      </div>
                    </List.Item>
                  )}
                />
              </div>
            ))}
          </div>
        </>
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
