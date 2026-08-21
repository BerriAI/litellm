import { Check, Copy } from "lucide-react";
import { useState, useCallback } from "react";
import moment from "moment";
import { AuditLogEntry, AUDIT_TABLE_NAME_DISPLAY } from "../AuditLogsTableColumns";
import DefaultProxyAdminTag from "../../common_components/DefaultProxyAdminTag";
import CopyButton from "@/components/shared/CopyButton";
import { StatusBadge, type StatusTone } from "@/components/shared/table_cells/status_badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";

interface AuditLogDrawerProps {
  open: boolean;
  onClose: () => void;
  log: AuditLogEntry | null;
}

const ACTION_TONE: Record<string, StatusTone> = {
  created: "success",
  updated: "info",
  deleted: "error",
  rotated: "warning",
};

function CopyableJsonBlock({ label, value }: { label: string; value: Record<string, any> }) {
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
    <div className="overflow-hidden rounded-sm border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border bg-muted px-3 py-2">
        <span className="text-xs font-semibold text-muted-foreground">{label}</span>
        <Button variant="ghost" size="icon-xs" onClick={handleCopy} title="Copy JSON" aria-label="Copy JSON">
          {copied ? <Check className="text-success" /> : <Copy />}
        </Button>
      </div>
      <pre className="m-0 max-h-96 overflow-auto bg-card p-3 font-mono text-xs break-all whitespace-pre-wrap">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

function MetadataRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 py-1.5">
      <span className="w-36 shrink-0 text-xs text-muted-foreground">{label}</span>
      <span className="text-xs break-all text-foreground">{value}</span>
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
    const changedBefore: Record<string, any> = {};
    const changedAfter: Record<string, any> = {};
    const allKeys = new Set([...Object.keys(before_value), ...Object.keys(updated_values)]);

    allKeys.forEach((key) => {
      const bStr = JSON.stringify(before_value[key]);
      const aStr = JSON.stringify(updated_values[key]);
      if (bStr !== aStr) {
        if (key in before_value) changedBefore[key] = before_value[key];
        if (key in updated_values) changedAfter[key] = updated_values[key];
      }
    });

    // Fields only in before (removed)
    Object.keys(before_value).forEach((key) => {
      if (!(key in updated_values) && !(key in changedBefore)) {
        changedBefore[key] = before_value[key];
        changedAfter[key] = undefined;
      }
    });

    // Fields only in after (added)
    Object.keys(updated_values).forEach((key) => {
      if (!(key in before_value) && !(key in changedAfter)) {
        changedAfter[key] = updated_values[key];
        changedBefore[key] = undefined;
      }
    });

    displayBefore = Object.keys(changedBefore).length > 0 ? changedBefore : { note: "No differing fields detected" };
    displayAfter = Object.keys(changedAfter).length > 0 ? changedAfter : { note: "No differing fields detected" };
  }

  const renderValue = (label: string, value: Record<string, any> | null | undefined) => {
    if (!value || Object.keys(value).length === 0) {
      return (
        <div className="overflow-hidden rounded-sm border border-border bg-card">
          <div className="flex items-center border-b border-border bg-muted px-3 py-2">
            <span className="text-xs font-semibold text-muted-foreground">{label}</span>
          </div>
          <p className="m-0 px-3 py-3 text-xs text-muted-foreground italic">N/A</p>
        </div>
      );
    }

    // For key table updates, show only meaningful fields as plain text
    if (isKeyTable && isUpdateAction) {
      const knownKeyFields = ["token", "spend", "max_budget"];
      const hasOnlyKnown = Object.keys(value).every((k) => knownKeyFields.includes(k));
      if (hasOnlyKnown && !("note" in value)) {
        return (
          <div className="overflow-hidden rounded-sm border border-border bg-card">
            <div className="flex items-center border-b border-border bg-muted px-3 py-2">
              <span className="text-xs font-semibold text-muted-foreground">{label}</span>
            </div>
            <div className="space-y-1 px-3 py-3 text-xs">
              {value.token !== undefined && (
                <p>
                  <span className="text-muted-foreground">Token:</span> {value.token ?? "N/A"}
                </p>
              )}
              {value.spend !== undefined && (
                <p>
                  <span className="text-muted-foreground">Spend:</span> ${Number(value.spend).toFixed(6)}
                </p>
              )}
              {value.max_budget !== undefined && (
                <p>
                  <span className="text-muted-foreground">Max Budget:</span> ${Number(value.max_budget).toFixed(6)}
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
    <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
      {renderValue("Before", displayBefore)}
      {renderValue("After", displayAfter)}
    </div>
  );
}

export function AuditLogDrawer({ open, onClose, log }: AuditLogDrawerProps) {
  if (!log) return null;

  const tableDisplay = AUDIT_TABLE_NAME_DISPLAY[log.table_name] ?? log.table_name;

  return (
    <Sheet open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <SheetContent side="right" className="w-[60%] gap-0 overflow-y-auto p-0 sm:max-w-none">
        <SheetTitle className="sr-only">Audit log details</SheetTitle>

        <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-6 py-4">
          <StatusBadge tone={ACTION_TONE[log.action] ?? "neutral"} label={log.action} />
          <span className="text-sm text-muted-foreground">
            {moment.utc(log.updated_at).local().format("MMM D, YYYY HH:mm:ss")}
          </span>
        </div>

        <div className="px-6 py-5">
          <div className="mb-5 rounded-lg border border-border bg-muted p-4">
            <p className="mb-2 text-xs font-semibold tracking-wide text-foreground uppercase">Details</p>
            <MetadataRow label="Table" value={tableDisplay} />
            <MetadataRow
              label="Object ID"
              value={
                <span className="inline-flex items-center gap-1 font-mono text-xs">
                  {log.object_id}
                  <CopyButton value={log.object_id} label="Copy object ID" />
                </span>
              }
            />
            <MetadataRow label="Changed By" value={<DefaultProxyAdminTag userId={log.changed_by} />} />
            <MetadataRow
              label="API Key (Hash)"
              value={
                log.changed_by_api_key ? (
                  <span className="inline-flex items-center gap-1 font-mono text-xs break-all">
                    {log.changed_by_api_key}
                    <CopyButton value={log.changed_by_api_key} label="Copy API key hash" />
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
