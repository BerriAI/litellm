import { Drawer, Tag, Typography } from "antd";
import { CloseOutlined, CopyOutlined, CheckOutlined } from "@ant-design/icons";
import { useState, useCallback } from "react";
import moment from "moment";
import { AuditLogEntry } from "../columns";
import DefaultProxyAdminTag from "../../common_components/DefaultProxyAdminTag";

const { Text } = Typography;

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

const ACTION_COLOR: Record<string, string> = {
  created: "green",
  updated: "blue",
  deleted: "red",
  rotated: "orange",
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
    <div className="bg-white rounded border overflow-hidden">
      <div className="flex justify-between items-center px-3 py-2 border-b bg-gray-50">
        <span className="text-xs font-semibold text-gray-600">{label}</span>
        <button
          onClick={handleCopy}
          className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700 transition-colors"
          title="Copy JSON"
        >
          {copied ? <CheckOutlined className="text-green-600" /> : <CopyOutlined />}
        </button>
      </div>
      <pre className="p-3 bg-white text-xs font-mono overflow-auto max-h-96 whitespace-pre-wrap break-all m-0">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

function MetadataRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 py-1.5">
      <span className="text-xs text-gray-500 w-36 shrink-0">{label}</span>
      <span className="text-xs text-gray-900 break-all">{value}</span>
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

    displayBefore =
      Object.keys(changedBefore).length > 0
        ? changedBefore
        : { note: "No differing fields detected" };
    displayAfter =
      Object.keys(changedAfter).length > 0
        ? changedAfter
        : { note: "No differing fields detected" };
  }

  const renderValue = (label: string, value: Record<string, any> | null | undefined) => {
    if (!value || Object.keys(value).length === 0) {
      return (
        <div className="bg-white rounded border overflow-hidden">
          <div className="flex items-center px-3 py-2 border-b bg-gray-50">
            <span className="text-xs font-semibold text-gray-600">{label}</span>
          </div>
          <p className="px-3 py-3 text-xs text-gray-400 italic m-0">N/A</p>
        </div>
      );
    }

    // For key table updates, show only meaningful fields as plain text
    if (isKeyTable && isUpdateAction) {
      const knownKeyFields = ["token", "spend", "max_budget"];
      const hasOnlyKnown = Object.keys(value).every((k) => knownKeyFields.includes(k));
      if (hasOnlyKnown && !("note" in value)) {
        return (
          <div className="bg-white rounded border overflow-hidden">
            <div className="flex items-center px-3 py-2 border-b bg-gray-50">
              <span className="text-xs font-semibold text-gray-600">{label}</span>
            </div>
            <div className="px-3 py-3 space-y-1 text-xs">
              {value.token !== undefined && (
                <p><span className="text-gray-500">Token:</span> {value.token ?? "N/A"}</p>
              )}
              {value.spend !== undefined && (
                <p><span className="text-gray-500">Spend:</span> ${Number(value.spend).toFixed(6)}</p>
              )}
              {value.max_budget !== undefined && (
                <p><span className="text-gray-500">Max Budget:</span> ${Number(value.max_budget).toFixed(6)}</p>
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
  const actionColor = ACTION_COLOR[log.action] ?? "default";

  return (
    <Drawer
      placement="right"
      width="60%"
      open={open}
      onClose={onClose}
      closable={false}
      mask={true}
      maskClosable={true}
      styles={{ body: { padding: 0, display: "flex", flexDirection: "column" }, header: { display: "none" } }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b bg-white shrink-0">
        <div className="flex items-center gap-3">
          <Tag color={actionColor} className="capitalize m-0">
            {log.action}
          </Tag>
          <span className="text-sm text-gray-500">
            {moment.utc(log.updated_at).local().format("MMM D, YYYY HH:mm:ss")}
          </span>
        </div>
        <button
          onClick={onClose}
          className="w-8 h-8 flex items-center justify-center rounded hover:bg-gray-100 text-gray-500"
          aria-label="Close"
        >
          <CloseOutlined />
        </button>
      </div>

      {/* Body */}
      <div className="px-6 py-5">
        {/* Metadata */}
        <div className="bg-gray-50 border rounded-lg p-4 mb-5">
          <p className="text-xs font-semibold text-gray-700 mb-2 uppercase tracking-wide">
            Details
          </p>
          <MetadataRow label="Table" value={tableDisplay} />
          <MetadataRow
            label="Object ID"
            value={
              <Text copyable className="font-mono text-xs">
                {log.object_id}
              </Text>
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
                <Text copyable className="font-mono text-xs break-all">
                  {log.changed_by_api_key}
                </Text>
              ) : (
                "—"
              )
            }
          />
        </div>

        {/* Diff */}
        <DiffSection log={log} />
      </div>
    </Drawer>
  );
}
