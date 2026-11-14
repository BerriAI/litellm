import React from "react";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
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
    return t("autoRotationView.timestamp", { date: dateStr, time: timeStr });
  };

  const content = (
    <div className="space-y-6">
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <RefreshIcon className="h-4 w-4 text-info" />
          <p className="text-sm font-semibold text-foreground">{t("autoRotationView.title")}</p>
          <StatusBadge
            tone={autoRotate ? "success" : "neutral"}
            label={autoRotate ? t("autoRotationView.enabled") : t("autoRotationView.disabled")}
          />
          {autoRotate && rotationInterval && (
            <>
              <p className="text-sm text-muted-foreground">•</p>
              <p className="text-sm text-muted-foreground">
                {t("autoRotationView.every", { interval: rotationInterval })}
              </p>
            </>
          )}
        </div>
      </div>

      {(autoRotate || lastRotationAt || keyRotationAt || nextRotationAt) && (
        <div className="space-y-3">
          {lastRotationAt && (
            <div className="flex items-center gap-2 rounded-md border border-border bg-muted p-3">
              <ClockIcon className="h-4 w-4 text-muted-foreground" />
              <div className="flex-1">
                <p className="text-sm font-medium text-foreground">{t("autoRotationView.lastRotation")}</p>
                <p className="text-sm text-muted-foreground">{formatTimestamp(lastRotationAt)}</p>
              </div>
            </div>
          )}

          {(keyRotationAt || nextRotationAt) && (
            <div className="flex items-center gap-2 rounded-md border border-border bg-muted p-3">
              <ClockIcon className="h-4 w-4 text-muted-foreground" />
              <div className="flex-1">
                <p className="text-sm font-medium text-foreground">{t("autoRotationView.nextScheduledRotation")}</p>
                <p className="text-sm text-muted-foreground">
                  {formatTimestamp(nextRotationAt || keyRotationAt || "")}
                </p>
              </div>
            </div>
          )}

          {autoRotate && !lastRotationAt && !keyRotationAt && !nextRotationAt && (
            <div className="flex items-center gap-2 rounded-md border border-border bg-muted p-3">
              <ClockIcon className="h-4 w-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">{t("autoRotationView.noHistory")}</p>
            </div>
          )}
        </div>
      )}

      {!autoRotate && !lastRotationAt && !keyRotationAt && !nextRotationAt && (
        <div className="flex items-center gap-2 rounded-md border border-border bg-muted p-3">
          <RefreshIcon className="h-4 w-4 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">{t("autoRotationView.notEnabled")}</p>
        </div>
      )}
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`rounded-lg border border-border bg-card p-6 ${className}`}>
        <div className="mb-6 flex items-center gap-2">
          <div>
            <p className="text-sm font-semibold text-foreground">{t("autoRotationView.title")}</p>
            <p className="text-xs text-muted-foreground">{t("autoRotationView.description")}</p>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <p className="mb-3 text-sm font-medium text-foreground">{t("autoRotationView.title")}</p>
      {content}
    </div>
  );
};

export default AutoRotationView;
