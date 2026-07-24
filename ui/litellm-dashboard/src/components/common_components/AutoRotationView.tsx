import React from "react";
import { Text, Badge } from "@tremor/react";
import { RefreshIcon, ClockIcon } from "@heroicons/react/outline";

interface AutoRotationViewProps {
  autoRotate?: boolean;
  rotationInterval?: string;
  lastRotationAt?: string;
  keyRotationAt?: string;
  nextRotationAt?: string;
  variant?: "card" | "inline";
  className?: string;
}

const AutoRotationView: React.FC<AutoRotationViewProps> = ({
  autoRotate = false,
  rotationInterval,
  lastRotationAt,
  keyRotationAt,
  nextRotationAt,
  variant = "card",
  className = "",
}) => {
  const formatTimestamp = (timestamp: string | Date) => {
    const date = new Date(timestamp);
    const dateStr = date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
    const timeStr = date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
    return `${dateStr} at ${timeStr}`;
  };

  const content = (
    <div className="space-y-6">
      {/* Status Section */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <RefreshIcon className="h-4 w-4 text-blue-600" />
          <Text className="font-semibold text-gray-900">Auto-Rotation</Text>
          <Badge color={autoRotate ? "green" : "gray"} size="xs">
            {autoRotate ? "Enabled" : "Disabled"}
          </Badge>
          {autoRotate && rotationInterval && (
            <>
              <Text className="text-gray-400">•</Text>
              <Text className="text-sm text-gray-600">Every {rotationInterval}</Text>
            </>
          )}
        </div>
      </div>

      {/* Rotation History - Show if there's any rotation data OR if auto-rotation is enabled */}
      {(autoRotate || lastRotationAt || keyRotationAt || nextRotationAt) && (
        <div className="space-y-3">
          {/* Last Rotation - Show when available */}
          {lastRotationAt && (
            <div className="flex items-center gap-2 p-3 bg-gray-50 border border-gray-200 rounded-md">
              <ClockIcon className="w-4 h-4 text-gray-500" />
              <div className="flex-1">
                <Text className="font-medium text-gray-700">Last Rotation</Text>
                <Text className="text-sm text-gray-600">{formatTimestamp(lastRotationAt)}</Text>
              </div>
            </div>
          )}

          {/* Next Scheduled Rotation - Show when available */}
          {(keyRotationAt || nextRotationAt) && (
            <div className="flex items-center gap-2 p-3 bg-gray-50 border border-gray-200 rounded-md">
              <ClockIcon className="w-4 h-4 text-gray-500" />
              <div className="flex-1">
                <Text className="font-medium text-gray-700">Next Scheduled Rotation</Text>
                <Text className="text-sm text-gray-600">{formatTimestamp(nextRotationAt || keyRotationAt || "")}</Text>
              </div>
            </div>
          )}

          {/* No rotation data message - Only show if auto-rotation is enabled but no data */}
          {autoRotate && !lastRotationAt && !keyRotationAt && !nextRotationAt && (
            <div className="flex items-center gap-2 p-3 bg-gray-50 border border-gray-100 rounded-md">
              <ClockIcon className="w-4 h-4 text-gray-500" />
              <Text className="text-gray-600">No rotation history available</Text>
            </div>
          )}
        </div>
      )}

      {/* Disabled State - Only show if auto-rotation is disabled AND there's no rotation history */}
      {!autoRotate && !lastRotationAt && !keyRotationAt && !nextRotationAt && (
        <div className="flex items-center gap-2 p-3 bg-gray-50 border border-gray-100 rounded-md">
          <RefreshIcon className="w-4 h-4 text-gray-400" />
          <Text className="text-gray-600">Auto-rotation is not enabled for this key</Text>
        </div>
      )}
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`bg-white border border-gray-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <div>
            <Text className="font-semibold text-gray-900">Auto-Rotation</Text>
            <Text className="text-xs text-gray-500">Automatic key rotation settings and status for this key</Text>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <Text className="font-medium text-gray-900 mb-3">Auto-Rotation</Text>
      {content}
    </div>
  );
};

export default AutoRotationView;
