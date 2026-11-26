import { copyToClipboard } from "@/utils/dataUtils";
import { useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, X } from "lucide-react";
import moment from "moment";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { LogEntry } from "../types";
import { AutoRouterTag } from "@/components/shared/table_cells";
import { ClassifyTag } from "./ClassifyTag";
import { SidebarToggle } from "./SidebarToggle";
import { getProviderLogoAndName } from "../../provider_info_helpers";

interface DrawerHeaderProps {
  log: LogEntry;
  onClose: () => void;
  onPrevious: () => void;
  onNext: () => void;
  statusLabel: string;
  statusColor: "error" | "success";
  environment: string;
  isSidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}

/**
 * Header component for the log details drawer.
 * Displays model/provider, request ID, navigation controls, status, environment, and timestamp.
 */
export function DrawerHeader({
  log,
  onClose,
  onPrevious,
  onNext,
  statusLabel,
  statusColor,
  environment,
  isSidebarCollapsed,
  onToggleSidebar,
}: DrawerHeaderProps) {
  const { t } = useTranslation();
  const provider = log.custom_llm_provider || "";
  const providerInfo = provider ? getProviderLogoAndName(provider) : null;
  const showToggleWithProvider = isSidebarCollapsed && Boolean(providerInfo || log.model);
  const showToggleWithRequestId = isSidebarCollapsed && !showToggleWithProvider;
  const callId: string | null =
    log.litellm_call_id && log.litellm_call_id !== log.request_id ? log.litellm_call_id : null;

  return (
    <div className="sticky top-0 z-chrome border-b border-border bg-background px-6 py-4">
      {/* Row 0: Model + Provider with Logo */}
      <div className="mb-2 flex items-center gap-2">
        {showToggleWithProvider && <SidebarToggle isCollapsed onToggle={onToggleSidebar} />}
        <ModelProviderSection
          model={log.model}
          modelGroup={log.model_group}
          internalCallOrigin={log.metadata?.internal_call_origin}
          providerLogo={providerInfo?.logo}
          providerName={providerInfo?.displayName}
        />
      </div>

      {/* Row 1: Request ID + Actions */}
      <div className="mb-2 flex items-center justify-between gap-1">
        {showToggleWithRequestId && <SidebarToggle isCollapsed onToggle={onToggleSidebar} />}
        <div className="min-w-0 flex-1">
          <CopyableId value={log.request_id} label={t("logs.detail.header.requestId")} />
          {callId && (
            <div className="flex items-center gap-1">
              <span className="text-[13px] whitespace-nowrap text-muted-foreground [font-family:monospace]">
                x-litellm-call-id:
              </span>
              <CopyableId value={callId} label="x-litellm-call-id" muted />
            </div>
          )}
        </div>
        <NavigationSection onPrevious={onPrevious} onNext={onNext} onClose={onClose} />
      </div>

      {/* Row 2: Status + Env + Timestamp */}
      <StatusBar log={log} statusLabel={statusLabel} statusColor={statusColor} environment={environment} />
    </div>
  );
}

/**
 * Model and Provider display with logo
 */
function ModelProviderSection({
  model,
  modelGroup,
  internalCallOrigin,
  providerLogo,
  providerName,
}: {
  model: string;
  modelGroup?: string;
  internalCallOrigin?: string | null;
  providerLogo?: string;
  providerName?: string;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex min-w-0 items-center gap-2">
      {providerLogo && (
        <img
          src={providerLogo}
          alt={providerName || t("logs.detail.header.provider")}
          className="size-6"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.style.display = "none";
          }}
        />
      )}
      <div className="flex items-center gap-2">
        <span className="text-[14px] font-semibold">
          {model}
        </span>
        {providerName && (
          <span className="text-[12px] text-muted-foreground">
            {providerName}
          </span>
        )}
        <AutoRouterTag modelGroup={modelGroup} />
        <ClassifyTag origin={internalCallOrigin} />
      </div>
    </div>
  );
}

function CopyableId({
  value,
  label,
  muted,
}: {
  value: string;
  label: string;
  muted?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const { t } = useTranslation();

  const handleCopy = async () => {
    if (await copyToClipboard(value)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger
          render={
            <span
              className={`block truncate [font-family:monospace] ${muted ? "text-[13px] text-muted-foreground" : "text-[16px] font-semibold"}`}
            />
          }
        >
          {value}
          <button
            type="button"
            aria-label={copied ? t("logs.detail.header.copied") : t("logs.detail.header.copyLabel", { label })}
            onClick={handleCopy}
            className="ml-1 align-middle text-muted-foreground hover:text-foreground"
          >
            {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          </button>
        </TooltipTrigger>
        <TooltipContent>{value}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Navigation controls (previous, next, close)
 * Shows keyboard shortcuts with bounding boxes for visibility
 */
function NavigationSection({
  onPrevious,
  onNext,
  onClose,
}: {
  onPrevious: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();

  return (
    <div className="flex items-center gap-1">
      <Button variant="ghost" size="sm" onClick={onPrevious}>
        <ChevronUp className="size-4" />
        <span className="ml-1 rounded-[4px] border border-border bg-muted px-1 text-[12px] [font-family:monospace]">
          K
        </span>
      </Button>
      <div className="h-5 w-px bg-border" />
      <Button variant="ghost" size="sm" onClick={onNext}>
        <ChevronDown className="size-4" />
        <span className="ml-1 rounded-[4px] border border-border bg-muted px-1 text-[12px] [font-family:monospace]">
          J
        </span>
      </Button>
      <div className="h-5 w-px bg-border" />
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger render={<Button variant="ghost" size="icon-sm" onClick={onClose} />}>
            <X className="size-4" />
          </TooltipTrigger>
          <TooltipContent>{t("logs.detail.header.escToClose")}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
}

/**
 * Status bar with tags and timestamp
 */
function StatusBar({
  log,
  statusLabel,
  statusColor,
  environment,
}: {
  log: LogEntry;
  statusLabel: string;
  statusColor: "error" | "success";
  environment: string;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex items-center gap-3">
      <Badge variant={statusColor === "error" ? "destructive" : "secondary"}>{statusLabel}</Badge>
      <Badge variant="outline">{t("logs.detail.header.environment", { environment })}</Badge>
      <div className="flex items-center gap-2">
        <span className="text-[13px] text-muted-foreground">
          {moment(log.startTime).format("MMM D, YYYY h:mm:ss A")}
        </span>
        <span className="text-[13px] text-muted-foreground">
          ({moment(log.startTime).fromNow()})
        </span>
      </div>
    </div>
  );
}
