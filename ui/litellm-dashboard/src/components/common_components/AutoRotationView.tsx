import React from "react";
import { StatusBadge } from "@/components/shared/table_cells";
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
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <RefreshIcon className="h-4 w-4 text-blue-600" />
          <p className="text-sm font-semibold text-gray-900">Auto-Rotation</p>
          <StatusBadge tone={autoRotate ? "success" : "neutral"} label={autoRotate ? "Enabled" : "Disabled"} />
          {autoRotate && rotationInterval && (
            <>
              <p className="text-sm text-gray-400">•</p>
              <p className="text-sm text-gray-600">Every {rotationInterval}</p>
            </>
          )}
        </div>
      </div>

      {(autoRotate || lastRotationAt || keyRotationAt || nextRotationAt) && (
        <div className="space-y-3">
          {lastRotationAt && (
            <div className="flex items-center gap-2 rounded-md border border-gray-200 bg-gray-50 p-3">
              <ClockIcon className="h-4 w-4 text-gray-500" />
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-700">Last Rotation</p>
                <p className="text-sm text-gray-600">{formatTimestamp(lastRotationAt)}</p>
              </div>
            </div>
          )}

          {(keyRotationAt || nextRotationAt) && (
            <div className="flex items-center gap-2 rounded-md border border-gray-200 bg-gray-50 p-3">
              <ClockIcon className="h-4 w-4 text-gray-500" />
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-700">Next Scheduled Rotation</p>
                <p className="text-sm text-gray-600">{formatTimestamp(nextRotationAt || keyRotationAt || "")}</p>
              </div>
            </div>
          )}

          {autoRotate && !lastRotationAt && !keyRotationAt && !nextRotationAt && (
            <div className="flex items-center gap-2 rounded-md border border-gray-100 bg-gray-50 p-3">
              <ClockIcon className="h-4 w-4 text-gray-500" />
              <p className="text-sm text-gray-600">No rotation history available</p>
            </div>
          )}
        </div>
      )}

      {!autoRotate && !lastRotationAt && !keyRotationAt && !nextRotationAt && (
        <div className="flex items-center gap-2 rounded-md border border-gray-100 bg-gray-50 p-3">
          <RefreshIcon className="h-4 w-4 text-gray-400" />
          <p className="text-sm text-gray-600">Auto-rotation is not enabled for this key</p>
        </div>
      )}
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`rounded-lg border border-gray-200 bg-white p-6 ${className}`}>
        <div className="mb-6 flex items-center gap-2">
          <div>
            <p className="text-sm font-semibold text-gray-900">Auto-Rotation</p>
            <p className="text-xs text-gray-500">Automatic key rotation settings and status for this key</p>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <p className="mb-3 text-sm font-medium text-gray-900">Auto-Rotation</p>
      {content}
    </div>
  );
};

export default AutoRotationView;
