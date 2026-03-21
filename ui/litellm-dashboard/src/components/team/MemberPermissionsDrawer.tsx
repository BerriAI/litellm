import { AvailablePermission, Member, teamMemberUpdateCall } from "@/components/networking";
import { CloudServerOutlined, SaveOutlined, UserOutlined } from "@ant-design/icons";
import { Alert, Button, Drawer, Switch, Tag, Typography } from "antd";
import React, { useEffect, useMemo, useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";

const { Text, Title: AntTitle } = Typography;

interface MemberPermissionsDrawerProps {
  open: boolean;
  onClose: () => void;
  member: Member | null;
  availablePermissions: AvailablePermission[];
  accessToken: string | null;
  teamId: string;
  onUpdate: () => void;
}

const MemberPermissionsDrawer: React.FC<MemberPermissionsDrawerProps> = ({
  open,
  onClose,
  member,
  availablePermissions,
  accessToken,
  teamId,
  onUpdate,
}) => {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);

  // Sync state when member changes or drawer opens
  useEffect(() => {
    if (open && member) {
      setSelected(new Set(member.extra_permissions || []));
    }
  }, [open, member]);

  const initial = useMemo(
    () => new Set(member?.extra_permissions || []),
    [member],
  );

  const isDirty = useMemo(() => {
    if (selected.size !== initial.size) return true;
    for (const p of selected) {
      if (!initial.has(p)) return true;
    }
    return false;
  }, [selected, initial]);

  const allSelected = availablePermissions.length > 0 && availablePermissions.every((p) => selected.has(p.value));

  const handleToggle = (value: string, checked: boolean) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(value);
      } else {
        next.delete(value);
      }
      return next;
    });
  };

  const handleToggleAll = (checked: boolean) => {
    if (checked) {
      setSelected(new Set(availablePermissions.map((p) => p.value)));
    } else {
      setSelected(new Set());
    }
  };

  const handleSave = async () => {
    if (!accessToken || !member) return;
    setSaving(true);
    try {
      const updatedMember: Member = {
        ...member,
        extra_permissions: Array.from(selected),
      };
      await teamMemberUpdateCall(accessToken, teamId, updatedMember);
      NotificationsManager.success("Permissions updated successfully");
      onUpdate();
      onClose();
    } catch (error: any) {
      const errMsg = error?.message || "Failed to update permissions";
      NotificationsManager.fromBackend(errMsg);
      console.error("Error updating permissions:", error);
    } finally {
      setSaving(false);
    }
  };

  const memberDisplay = member?.user_email || member?.user_id || "Unknown";
  const isAdmin = member?.role?.toLowerCase() === "admin";

  return (
    <Drawer
      title={
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 bg-blue-50 text-blue-600 rounded-full shrink-0">
            <UserOutlined />
          </div>
          <div className="flex flex-col overflow-hidden">
            <Text strong className="text-base leading-tight truncate">
              {memberDisplay}
            </Text>
            <Text type="secondary" className="text-xs truncate">
              {member?.user_id || ""}
            </Text>
          </div>
          <Tag color={isAdmin ? "gold" : "default"} className="ml-auto shrink-0">
            {member?.role || "user"}
          </Tag>
        </div>
      }
      placement="right"
      width={480}
      open={open}
      onClose={onClose}
      closable={!saving}
      maskClosable={!saving}
      footer={
        <div className="flex justify-end gap-2">
          <Button onClick={onClose} disabled={saving}>Cancel</Button>
          <Button
            type="primary"
            icon={<SaveOutlined />}
            onClick={handleSave}
            loading={saving}
            disabled={!isDirty}
          >
            Save Changes
          </Button>
        </div>
      }
    >
      <div className="flex flex-col gap-6">
        {isAdmin ? (
          <Alert
            message="Team admins have all permissions by default. Extra permissions only apply to non-admin members."
            type="info"
            showIcon
          />
        ) : (
          <Alert
            message="Permissions allow this member to manage MCP servers for this team."
            type="info"
            showIcon
          />
        )}

        {/* MCP Server Permissions */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <CloudServerOutlined className="text-lg text-gray-600" />
            <AntTitle level={5} className="!m-0">
              MCP Server Permissions
            </AntTitle>
          </div>

          <div className="bg-gray-50 p-3 rounded-lg border border-gray-200 mb-4 flex items-center justify-between">
            <Text strong>{allSelected ? "Deselect All" : "Select All"}</Text>
            <Switch
              checked={allSelected}
              onChange={handleToggleAll}
              disabled={isAdmin}
            />
          </div>

          <div className="flex flex-col">
            {availablePermissions.map((perm) => (
              <div
                key={perm.value}
                className="flex items-center justify-between py-3 border-b border-gray-100 last:border-0"
              >
                <div className="flex flex-col gap-1">
                  <Text>{perm.label}</Text>
                  <Text
                    type="secondary"
                    className="font-mono text-xs bg-gray-100 px-1.5 py-0.5 rounded w-fit border border-gray-200"
                  >
                    {perm.value}
                  </Text>
                </div>
                <Switch
                  checked={selected.has(perm.value)}
                  onChange={(checked) => handleToggle(perm.value, checked)}
                  disabled={isAdmin}
                />
              </div>
            ))}
          </div>
        </div>

        {availablePermissions.length === 0 && (
          <div className="text-center py-8 text-gray-400 text-sm">
            No permissions available.
          </div>
        )}
      </div>
    </Drawer>
  );
};

export default MemberPermissionsDrawer;
