import React from "react";
import { Card, Text, Badge } from "@tremor/react";
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
    <div className={className}>
      <div className="flex items-center gap-2 mb-4">
        <RefreshIcon className="w-5 h-5 text-blue-500" />
        <Text className="font-medium text-lg">Auto-Rotation</Text>
      </div>
      
      <Text className="text-gray-600 text-sm mb-4">
        Automatic key rotation settings and status for this key
      </Text>

      <div className="space-y-4">
        {/* Status Section */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 p-3 bg-gray-50 border border-gray-100 rounded-md w-full">
            <RefreshIcon className="w-4 h-4 text-gray-500" />
            <Text className="font-medium">Status:</Text>
            <Badge color={autoRotate ? "green" : "gray"} size="xs">
              {autoRotate ? "Enabled" : "Disabled"}
            </Badge>
            {autoRotate && rotationInterval && (
              <>
                <Text className="text-gray-400">•</Text>
                <Text className="text-sm">Every {rotationInterval}</Text>
              </>
            )}
          </div>
        </div>

        {/* Rotation History - Show if there's any rotation data OR if auto-rotation is enabled */}
        {(autoRotate || lastRotationAt || keyRotationAt || nextRotationAt) && (
          <div className="space-y-2">
            {lastRotationAt && (
              <div className="flex items-center gap-2 p-3 bg-blue-50 border border-blue-100 rounded-md">
                <ClockIcon className="w-4 h-4 text-blue-500" />
                <div className="flex-1">
                  <Text className="font-medium text-blue-700">Last Rotation</Text>
                  <Text className="text-sm text-blue-600">{formatTimestamp(lastRotationAt)}</Text>
                </div>
              </div>
            )}

            {(keyRotationAt || nextRotationAt) && (
              <div className="flex items-center gap-2 p-3 bg-orange-50 border border-orange-100 rounded-md">
                <ClockIcon className="w-4 h-4 text-orange-500" />
                <div className="flex-1">
                  <Text className="font-medium text-orange-700">Next Scheduled Rotation</Text>
                  <Text className="text-sm text-orange-600">
                    {formatTimestamp(nextRotationAt || keyRotationAt || "")}
                  </Text>
                </div>
              </div>
            )}

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
    </div>
  );

  if (variant === "card") {
    return <Card>{content}</Card>;
  }

  return content;
};

export default AutoRotationView;
