import { useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, X } from "lucide-react";
import moment from "moment";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { LogEntry } from "../columns";
import { AutoRouterTag } from "@/components/shared/table_cells";
import { ClassifyTag } from "./ClassifyTag";
import { getProviderLogoAndName } from "../../provider_info_helpers";
import {
  DRAWER_HEADER_PADDING,
  COLOR_BORDER,
  COLOR_BACKGROUND,
  SPACING_MEDIUM,
  FONT_SIZE_HEADER,
  FONT_SIZE_MEDIUM,
  FONT_FAMILY_MONO,
} from "./constants";

interface DrawerHeaderProps {
  log: LogEntry;
  onClose: () => void;
  onPrevious: () => void;
  onNext: () => void;
  statusLabel: string;
  statusColor: "error" | "success";
  environment: string;
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
}: DrawerHeaderProps) {
  const provider = log.custom_llm_provider || "";
  const providerInfo = provider ? getProviderLogoAndName(provider) : null;

  return (
    <div
      className="z-chrome"
      style={{
        padding: DRAWER_HEADER_PADDING,
        borderBottom: `1px solid ${COLOR_BORDER}`,
        backgroundColor: COLOR_BACKGROUND,
        position: "sticky",
        top: 0,
      }}
    >
      {/* Row 0: Model + Provider with Logo */}
      <ModelProviderSection
        model={log.model}
        modelGroup={log.model_group}
        internalCallOrigin={log.metadata?.internal_call_origin}
        providerLogo={providerInfo?.logo}
        providerName={providerInfo?.displayName}
      />

      {/* Row 1: Request ID + Actions */}
      <div
        style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: SPACING_MEDIUM }}
      >
        <RequestIdSection requestId={log.request_id} />
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
  return (
    <div className="flex items-center gap-2" style={{ marginBottom: SPACING_MEDIUM }}>
      {providerLogo && (
        <img
          src={providerLogo}
          alt={providerName || "Provider"}
          style={{ width: 24, height: 24 }}
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.style.display = "none";
          }}
        />
      )}
      <div className="flex items-center gap-2">
        <span className="font-semibold" style={{ fontSize: 14 }}>
          {model}
        </span>
        {providerName && (
          <span className="text-muted-foreground" style={{ fontSize: 12 }}>
            {providerName}
          </span>
        )}
        <AutoRouterTag modelGroup={modelGroup} />
        <ClassifyTag origin={internalCallOrigin} />
      </div>
    </div>
  );
}

/**
 * Request ID display with copy functionality
 */
function RequestIdSection({ requestId }: { requestId: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(requestId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable in non-secure contexts */
    }
  };

  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            render={
              <span
                className="font-semibold"
                style={{
                  fontSize: FONT_SIZE_HEADER,
                  fontFamily: FONT_FAMILY_MONO,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  display: "block",
                }}
              />
            }
          >
            {requestId}
            <button
              type="button"
              aria-label={copied ? "Copied!" : "Copy Request ID"}
              onClick={handleCopy}
              className="ml-1 align-middle text-muted-foreground hover:text-foreground"
            >
              {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
            </button>
          </TooltipTrigger>
          <TooltipContent>{requestId}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
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
  const keyboardShortcutStyle = {
    border: "1px solid var(--color-border)",
    borderRadius: 4,
    padding: "0 4px",
    fontSize: 12,
    fontFamily: "monospace",
    marginLeft: 4,
    background: "var(--color-muted)",
  };
  const splitStyle = { width: 1, height: 20, background: COLOR_BORDER };

  return (
    <div className="flex items-center gap-1">
      <Button variant="ghost" size="sm" onClick={onPrevious}>
        <ChevronUp className="size-4" />
        <span style={keyboardShortcutStyle}>K</span>
      </Button>
      <div style={splitStyle} />
      <Button variant="ghost" size="sm" onClick={onNext}>
        <ChevronDown className="size-4" />
        <span style={keyboardShortcutStyle}>J</span>
      </Button>
      <div style={splitStyle} />
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger render={<Button variant="ghost" size="icon-sm" onClick={onClose} />}>
            <X className="size-4" />
          </TooltipTrigger>
          <TooltipContent>ESC to close</TooltipContent>
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
  return (
    <div className="flex items-center gap-3">
      <Badge variant={statusColor === "error" ? "destructive" : "secondary"}>{statusLabel}</Badge>
      <Badge variant="outline">Env: {environment}</Badge>
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground" style={{ fontSize: FONT_SIZE_MEDIUM }}>
          {moment(log.startTime).format("MMM D, YYYY h:mm:ss A")}
        </span>
        <span className="text-muted-foreground" style={{ fontSize: FONT_SIZE_MEDIUM }}>
          ({moment(log.startTime).fromNow()})
        </span>
      </div>
    </div>
  );
}
