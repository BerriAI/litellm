import {
  Sheet,
  SheetContent,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Check, Copy, X } from "lucide-react";
import { useState, useCallback } from "react";
import moment from "moment";
import { AuditLogEntry } from "../columns";
import DefaultProxyAdminTag from "../../common_components/DefaultProxyAdminTag";

interface AuditLogDrawerProps {
  open: boolean;
  onClose: () => void;
  log: AuditLogEntry | null;
}

const TABLE_NAME_DISPLAY: Record<string, string> = {
  LiteLLM_VerificationToken: "Keys",
  LiteLLM_TeamTable: "Teams",
  LiteLLM_UserTable: "Users",
  LiteLLM_OrganizationTable: "Organizations",
  LiteLLM_ProxyModelTable: "Models",
};

const ACTION_BADGE_CLASSES: Record<string, string> = {
  created:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  updated: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  deleted: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  rotated:
    "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
};

function CopyableText({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(value);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("Copy failed:", e);
    }
  }, [value]);
  return (
    <span className="inline-flex items-center gap-1">
      <span>{value}</span>
      <button
        type="button"
        onClick={handleCopy}
        className="text-muted-foreground hover:text-foreground"
        aria-label="Copy"
      >
        {copied ? (
          <Check className="h-3 w-3 text-emerald-500" />
        ) : (
          <Copy className="h-3 w-3" />
        )}
      </button>
    </span>
  );
}

function CopyableJsonBlock({
  label,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value,
}: {
  label: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value: Record<string, any>;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      const text = JSON.stringify(value, null, 2);
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const el = document.createElement("textarea");
        el.value = text;
        el.style.position = "fixed";
        el.style.opacity = "0";
        document.body.appendChild(el);
        el.focus();
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error("Copy failed:", e);
    }
  }, [value]);

  return (
    <div className="bg-background rounded border border-border overflow-hidden">
      <div className="flex justify-between items-center px-3 py-2 border-b border-border bg-muted">
        <span className="text-xs font-semibold text-muted-foreground">
          {label}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="p-1 hover:bg-muted-foreground/10 rounded text-muted-foreground hover:text-foreground transition-colors"
          title="Copy JSON"
          aria-label="Copy JSON"
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 text-emerald-600" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
        </button>
      </div>
      <pre className="p-3 bg-background text-xs font-mono overflow-auto max-h-96 whitespace-pre-wrap break-all m-0">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

function MetadataRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-start gap-2 py-1.5">
      <span className="text-xs text-muted-foreground w-36 shrink-0">
        {label}
      </span>
      <span className="text-xs text-foreground break-all">{value}</span>
    </div>
  );
}

function DiffSection({ log }: { log: AuditLogEntry }) {
  const { action, table_name, before_value, updated_values } = log;
  const isKeyTable = table_name === "LiteLLM_VerificationToken";
  const isUpdateAction = action === "updated" || action === "rotated";

  let displayBefore = before_value;
  let displayAfter = updated_values;

  if (isUpdateAction && before_value && updated_values) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const changedBefore: Record<string, any> = {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const changedAfter: Record<string, any> = {};
    const allKeys = new Set([
      ...Object.keys(before_value),
      ...Object.keys(updated_values),
    ]);

    allKeys.forEach((key) => {
      const bStr = JSON.stringify(before_value[key]);
      const aStr = JSON.stringify(updated_values[key]);
      if (bStr !== aStr) {
        if (key in before_value) changedBefore[key] = before_value[key];
        if (key in updated_values) changedAfter[key] = updated_values[key];
      }
    });

    Object.keys(before_value).forEach((key) => {
      if (!(key in updated_values) && !(key in changedBefore)) {
        changedBefore[key] = before_value[key];
        changedAfter[key] = undefined;
      }
    });

    Object.keys(updated_values).forEach((key) => {
      if (!(key in before_value) && !(key in changedAfter)) {
        changedAfter[key] = updated_values[key];
        changedBefore[key] = undefined;
      }
    });

    displayBefore =
      Object.keys(changedBefore).length > 0
        ? changedBefore
        : { note: "No differing fields detected" };
    displayAfter =
      Object.keys(changedAfter).length > 0
        ? changedAfter
        : { note: "No differing fields detected" };
  }

  const renderValue = (
    label: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    value: Record<string, any> | null | undefined,
  ) => {
    if (!value || Object.keys(value).length === 0) {
      return (
        <div className="bg-background rounded border border-border overflow-hidden">
          <div className="flex items-center px-3 py-2 border-b border-border bg-muted">
            <span className="text-xs font-semibold text-muted-foreground">
              {label}
            </span>
          </div>
          <p className="px-3 py-3 text-xs text-muted-foreground italic m-0">
            N/A
          </p>
        </div>
      );
    }

    if (isKeyTable && isUpdateAction) {
      const knownKeyFields = ["token", "spend", "max_budget"];
      const hasOnlyKnown = Object.keys(value).every((k) =>
        knownKeyFields.includes(k),
      );
      if (hasOnlyKnown && !("note" in value)) {
        return (
          <div className="bg-background rounded border border-border overflow-hidden">
            <div className="flex items-center px-3 py-2 border-b border-border bg-muted">
              <span className="text-xs font-semibold text-muted-foreground">
                {label}
              </span>
            </div>
            <div className="px-3 py-3 space-y-1 text-xs">
              {value.token !== undefined && (
                <p>
                  <span className="text-muted-foreground">Token:</span>{" "}
                  {value.token ?? "N/A"}
                </p>
              )}
              {value.spend !== undefined && (
                <p>
                  <span className="text-muted-foreground">Spend:</span> $
                  {Number(value.spend).toFixed(6)}
                </p>
              )}
              {value.max_budget !== undefined && (
                <p>
                  <span className="text-muted-foreground">Max Budget:</span> $
                  {Number(value.max_budget).toFixed(6)}
                </p>
              )}
            </div>
          </div>
        );
      }
    }

    return <CopyableJsonBlock label={label} value={value} />;
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
      {renderValue("Before", displayBefore)}
      {renderValue("After", displayAfter)}
    </div>
  );
}

export function AuditLogDrawer({ open, onClose, log }: AuditLogDrawerProps) {
  if (!log) return null;

  const tableDisplay = TABLE_NAME_DISPLAY[log.table_name] ?? log.table_name;
  const actionClasses =
    ACTION_BADGE_CLASSES[log.action] ?? "bg-muted text-muted-foreground";

  return (
    <Sheet open={open} onOpenChange={(o) => (!o ? onClose() : undefined)}>
      <SheetContent
        side="right"
        className="w-[60%] sm:max-w-[60%] p-0 flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-background shrink-0">
          <div className="flex items-center gap-3">
            <Badge className={cn("capitalize", actionClasses)}>
              {log.action}
            </Badge>
            <span className="text-sm text-muted-foreground">
              {moment
                .utc(log.updated_at)
                .local()
                .format("MMM D, YYYY HH:mm:ss")}
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded hover:bg-muted text-muted-foreground"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 overflow-auto">
          <div className="bg-muted border border-border rounded-lg p-4 mb-5">
            <p className="text-xs font-semibold text-foreground mb-2 uppercase tracking-wide">
              Details
            </p>
            <MetadataRow label="Table" value={tableDisplay} />
            <MetadataRow
              label="Object ID"
              value={
                <span className="font-mono text-xs">
                  <CopyableText value={log.object_id} />
                </span>
              }
            />
            <MetadataRow
              label="Changed By"
              value={<DefaultProxyAdminTag userId={log.changed_by} />}
            />
            <MetadataRow
              label="API Key (Hash)"
              value={
                log.changed_by_api_key ? (
                  <span className="font-mono text-xs break-all">
                    <CopyableText value={log.changed_by_api_key} />
                  </span>
                ) : (
                  "—"
                )
              }
            />
          </div>

          <DiffSection log={log} />
        </div>
      </SheetContent>
    </Sheet>
  );
}
