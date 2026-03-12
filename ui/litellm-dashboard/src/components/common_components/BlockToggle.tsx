import React, { useState, useEffect } from "react";
import { Badge, Text } from "@tremor/react";
import { Switch, message } from "antd";
import { LockOutlined, UnlockOutlined } from "@ant-design/icons";
import { getProxyBaseUrl } from "../networking";

export interface BlockToggleProps {
  entityType: "user" | "team" | "key";
  entityId: string;
  currentBlockedStatus: boolean;
  onToggle?: (newStatus: boolean) => Promise<void>;
  accessToken: string;
  baseUrl?: string;
  disabled?: boolean;
  userRole?: string | null;
}

const BlockToggle: React.FC<BlockToggleProps> = ({
  entityType,
  entityId,
  currentBlockedStatus,
  onToggle,
  accessToken,
  baseUrl,
  disabled = false,
  userRole = null,
}) => {
  const [isBlocked, setIsBlocked] = useState(currentBlockedStatus);
  const [isLoading, setIsLoading] = useState(false);

  // Sync state when prop changes externally
  useEffect(() => {
    setIsBlocked(currentBlockedStatus);
  }, [currentBlockedStatus]);

  // Only render for proxy admins
  if (userRole !== "proxy_admin") {
    return null;
  }

  const handleToggle = async (checked: boolean) => {
    // Prevent toggle when disabled
    if (disabled) return;

    // checked = true means Active (not blocked)
    // checked = false means Blocked
    const newBlockedStatus = !checked;

    setIsLoading(true);
    try {
      const effectiveBaseUrl = baseUrl || getProxyBaseUrl();
      const endpoint = newBlockedStatus
        ? `${effectiveBaseUrl}/${entityType}/block`
        : `${effectiveBaseUrl}/${entityType}/unblock`;

      // For keys, the API expects "key" not "key_id"
      const paramName = entityType === "key" ? "key" : `${entityType}_id`;
      
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ [paramName]: entityId }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail || errorData.error || `Failed to ${newBlockedStatus ? "block" : "unblock"} ${entityType}`
        );
      }

      setIsBlocked(newBlockedStatus);

      // Call the optional onToggle callback
      if (onToggle) {
        await onToggle(newBlockedStatus);
      }

      message.success(
        `${entityType.charAt(0).toUpperCase() + entityType.slice(1)} ${newBlockedStatus ? "blocked" : "unblocked"} successfully`
      );
    } catch (error) {
      console.error(`Error toggling ${entityType} block status:`, error);
      message.error(
        error instanceof Error
          ? error.message
          : `Failed to ${newBlockedStatus ? "block" : "unblock"} ${entityType}`
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        {isBlocked ? (
          <LockOutlined className="text-red-500" />
        ) : (
          <UnlockOutlined className="text-green-500" />
        )}
        <Text className="font-medium">Status:</Text>
      </div>
      <Switch
        checked={!isBlocked}
        onChange={handleToggle}
        loading={isLoading}
        disabled={disabled || isLoading}
        checkedChildren="Active"
        unCheckedChildren="Blocked"
        className={disabled ? "opacity-50 cursor-not-allowed" : ""}
      />
      {isBlocked && (
        <Badge color="red" size="xs">
          BLOCKED
        </Badge>
      )}
    </div>
  );
};

export default BlockToggle;
