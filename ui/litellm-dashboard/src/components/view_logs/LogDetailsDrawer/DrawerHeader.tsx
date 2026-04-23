import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Check, ChevronDown, ChevronUp, Copy, X } from "lucide-react";
import moment from "moment";
import { LogEntry } from "../columns";
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
      className="sticky top-0 z-10 border-b"
      style={{
        padding: DRAWER_HEADER_PADDING,
        borderBottomColor: COLOR_BORDER,
        backgroundColor: COLOR_BACKGROUND,
      }}
    >
      <ModelProviderSection
        model={log.model}
        providerLogo={providerInfo?.logo}
        providerName={providerInfo?.displayName}
      />

      <div
        className="flex items-center justify-between"
        style={{ marginBottom: SPACING_MEDIUM }}
      >
        <RequestIdSection requestId={log.request_id} />
        <NavigationSection
          onPrevious={onPrevious}
          onNext={onNext}
          onClose={onClose}
        />
      </div>

      <StatusBar
        log={log}
        statusLabel={statusLabel}
        statusColor={statusColor}
        environment={environment}
      />
    </div>
  );
}

function ModelProviderSection({
  model,
  providerLogo,
  providerName,
}: {
  model: string;
  providerLogo?: string;
  providerName?: string;
}) {
  return (
    <div className="flex items-center gap-3 mb-3">
      {providerLogo && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={providerLogo}
          alt={providerName || "Provider"}
          className="w-6 h-6"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.style.display = "none";
          }}
        />
      )}
      <div className="flex items-center gap-3">
        <span className="font-bold text-sm">{model}</span>
        {providerName && (
          <span className="text-xs text-muted-foreground">{providerName}</span>
        )}
      </div>
    </div>
  );
}

function RequestIdSection({ requestId }: { requestId: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(requestId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_err) {
      /* noop */
    }
  };

  return (
    <div className="flex-1 min-w-0 flex items-center gap-1">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="block overflow-hidden text-ellipsis whitespace-nowrap font-bold"
              style={{
                fontSize: FONT_SIZE_HEADER,
                fontFamily: FONT_FAMILY_MONO,
              }}
            >
              {requestId}
            </span>
          </TooltipTrigger>
          <TooltipContent>{requestId}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <TooltipProvider>
        <Tooltip open={copied || undefined}>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={handleCopy}
              className="text-muted-foreground hover:text-foreground shrink-0"
              aria-label="Copy Request ID"
            >
              {copied ? (
                <Check className="h-3 w-3 text-emerald-500" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent>
            {copied ? "Copied!" : "Copy Request ID"}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
}

function NavigationSection({
  onPrevious,
  onNext,
  onClose,
}: {
  onPrevious: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  const KeyHint = ({ children }: { children: React.ReactNode }) => (
    <span className="border border-border rounded px-1 text-xs font-mono ml-1 bg-muted">
      {children}
    </span>
  );

  return (
    <div className="flex items-center gap-1 divide-x divide-border">
      <div className="flex items-center gap-1 pr-1">
        <Button variant="ghost" size="sm" onClick={onPrevious}>
          <ChevronUp className="h-3.5 w-3.5" />
          <KeyHint>K</KeyHint>
        </Button>
      </div>
      <div className="flex items-center gap-1 px-1">
        <Button variant="ghost" size="sm" onClick={onNext}>
          <ChevronDown className="h-3.5 w-3.5" />
          <KeyHint>J</KeyHint>
        </Button>
      </div>
      <div className="flex items-center gap-1 pl-1">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-7 w-7"
                aria-label="Close"
              >
                <X className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>ESC to close</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}

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
    <div className="flex items-center gap-4">
      <Badge
        className={cn(
          statusColor === "error"
            ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
            : "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
        )}
      >
        {statusLabel}
      </Badge>
      <Badge variant="secondary">Env: {environment}</Badge>
      <div className="flex items-center gap-3">
        <span
          className="text-muted-foreground"
          style={{ fontSize: FONT_SIZE_MEDIUM }}
        >
          {moment(log.startTime).format("MMM D, YYYY h:mm:ss A")}
        </span>
        <span
          className="text-muted-foreground"
          style={{ fontSize: FONT_SIZE_MEDIUM }}
        >
          ({moment(log.startTime).fromNow()})
        </span>
      </div>
    </div>
  );
}
