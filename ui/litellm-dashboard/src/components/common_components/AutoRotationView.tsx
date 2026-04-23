import React from "react";
import { Badge } from "@/components/ui/badge";
import { Clock, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";

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
          <RefreshCw className="h-4 w-4 text-primary" />
          <span className="font-semibold text-foreground">Auto-Rotation</span>
          <Badge
            className={cn(
              autoRotate
                ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                : "bg-muted text-muted-foreground",
            )}
          >
            {autoRotate ? "Enabled" : "Disabled"}
          </Badge>
          {autoRotate && rotationInterval && (
            <>
              <span className="text-muted-foreground">•</span>
              <span className="text-sm text-muted-foreground">
                Every {rotationInterval}
              </span>
            </>
          )}
        </div>
      </div>

      {(autoRotate || lastRotationAt || keyRotationAt || nextRotationAt) && (
        <div className="space-y-3">
          {lastRotationAt && (
            <div className="flex items-center gap-2 p-3 bg-muted border border-border rounded-md">
              <Clock className="w-4 h-4 text-muted-foreground" />
              <div className="flex-1">
                <div className="font-medium text-foreground">Last Rotation</div>
                <div className="text-sm text-muted-foreground">
                  {formatTimestamp(lastRotationAt)}
                </div>
              </div>
            </div>
          )}

          {(keyRotationAt || nextRotationAt) && (
            <div className="flex items-center gap-2 p-3 bg-muted border border-border rounded-md">
              <Clock className="w-4 h-4 text-muted-foreground" />
              <div className="flex-1">
                <div className="font-medium text-foreground">
                  Next Scheduled Rotation
                </div>
                <div className="text-sm text-muted-foreground">
                  {formatTimestamp(nextRotationAt || keyRotationAt || "")}
                </div>
              </div>
            </div>
          )}

          {autoRotate &&
            !lastRotationAt &&
            !keyRotationAt &&
            !nextRotationAt && (
              <div className="flex items-center gap-2 p-3 bg-muted border border-border rounded-md">
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-muted-foreground">
                  No rotation history available
                </span>
              </div>
            )}
        </div>
      )}

      {!autoRotate &&
        !lastRotationAt &&
        !keyRotationAt &&
        !nextRotationAt && (
          <div className="flex items-center gap-2 p-3 bg-muted border border-border rounded-md">
            <RefreshCw className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">
              Auto-rotation is not enabled for this key
            </span>
          </div>
        )}
    </div>
  );

  if (variant === "card") {
    return (
      <div
        className={`bg-background border border-border rounded-lg p-6 ${className}`}
      >
        <div className="flex items-center gap-2 mb-6">
          <div>
            <p className="font-semibold text-foreground">Auto-Rotation</p>
            <p className="text-xs text-muted-foreground">
              Automatic key rotation settings and status for this key
            </p>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <p className="font-medium text-foreground mb-3">Auto-Rotation</p>
      {content}
    </div>
  );
};

export default AutoRotationView;
