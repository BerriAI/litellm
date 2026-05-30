"use client";

import React, { useEffect, useState } from "react";
import { Input, Modal, Tag, Typography } from "antd";
import {
  CheckOutlined,
  CloseOutlined,
  LockOutlined,
} from "@ant-design/icons";
import MessageManager from "@/components/molecules/message_manager";
import { MCPServer, MCPUserField, MCPUserFieldsStatus } from "./types";

interface UserFieldsModalProps {
  server: MCPServer;
  open: boolean;
  onClose: () => void;
  onSuccess: (status: MCPUserFieldsStatus) => void;
  accessToken: string;
}

/**
 * Dashboard modal where an end-user fills in the per-user fields declared by
 * the admin for an MCP server. On submit, posts the values to
 * /v1/mcp/server/{server_id}/user-field-values and returns the new status so
 * the caller can clear the red badge.
 */
export const UserFieldsModal: React.FC<UserFieldsModalProps> = ({
  server,
  open,
  onClose,
  onSuccess,
  accessToken,
}) => {
  const [values, setValues] = useState<Record<string, string>>({});
  const [storedFieldKeys, setStoredFieldKeys] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(false);

  const declaredFields: MCPUserField[] = server.user_fields ?? [];
  const displayName = server.alias || server.server_name || "Service";

  // When the modal opens, fetch the current status so the user can see which
  // fields they've already saved (we don't echo back the values themselves).
  useEffect(() => {
    if (!open || !accessToken) return;
    let cancelled = false;
    const load = async () => {
      setFetching(true);
      try {
        const res = await fetch(
          `/v1/mcp/server/${server.server_id}/user-field-values`,
          {
            method: "GET",
            headers: { Authorization: `Bearer ${accessToken}` },
          },
        );
        if (!res.ok) return;
        const status = (await res.json()) as MCPUserFieldsStatus;
        if (!cancelled) {
          setStoredFieldKeys(new Set(status.stored_field_keys ?? []));
        }
      } catch (err) {
        // Non-fatal; just leave storedFieldKeys empty.
      } finally {
        if (!cancelled) setFetching(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [open, server.server_id, accessToken]);

  const handleClose = () => {
    setValues({});
    setLoading(false);
    onClose();
  };

  const handleSave = async () => {
    // Trim and drop empty strings so optional fields aren't overwritten with "".
    const payloadValues: Record<string, string> = {};
    for (const field of declaredFields) {
      const v = values[field.field_key];
      if (typeof v === "string" && v.trim().length > 0) {
        payloadValues[field.field_key] = v.trim();
      }
    }
    // Validate required fields the user hasn't already saved.
    const missingRequired = declaredFields.filter(
      (f) =>
        (f.required ?? true) &&
        !storedFieldKeys.has(f.field_key) &&
        !payloadValues[f.field_key],
    );
    if (missingRequired.length > 0) {
      MessageManager.error(
        `Please fill in: ${missingRequired
          .map((f) => f.display_name || f.field_key)
          .join(", ")}`,
      );
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(
        `/v1/mcp/server/${server.server_id}/user-field-values`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          body: JSON.stringify({ values: payloadValues }),
        },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail?.error || err?.detail || "Failed to save fields");
      }
      const status = (await res.json()) as MCPUserFieldsStatus;
      MessageManager.success(`Saved your fields for ${displayName}`);
      onSuccess(status);
      handleClose();
    } catch (e: any) {
      MessageManager.error(e?.message || "Failed to save fields");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      open={open}
      onCancel={handleClose}
      footer={null}
      width={520}
      closeIcon={null}
    >
      <div className="relative p-2">
        <div className="flex items-center justify-between mb-4">
          <Typography.Title level={4} className="!mb-0">
            Connect {displayName}
          </Typography.Title>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600"
            aria-label="Close"
          >
            <CloseOutlined />
          </button>
        </div>

        <Typography.Paragraph type="secondary" className="!mb-4">
          This server needs a few personal values from you before you can use it.
          Your values are encrypted at rest and only sent to {displayName} on
          your behalf.
        </Typography.Paragraph>

        {declaredFields.length === 0 ? (
          <Typography.Text type="secondary">
            This server has no user fields to configure.
          </Typography.Text>
        ) : (
          <div className="space-y-4">
            {declaredFields.map((field) => {
              const isSaved = storedFieldKeys.has(field.field_key);
              const required = field.required ?? true;
              return (
                <div key={field.field_key}>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-sm font-semibold text-gray-800">
                      {field.display_name || field.field_key}
                      {required && (
                        <span className="text-red-500 ml-1" title="Required">
                          *
                        </span>
                      )}
                    </label>
                    {isSaved && (
                      <Tag color="green" className="!text-xs !mr-0">
                        <CheckOutlined /> Saved
                      </Tag>
                    )}
                  </div>
                  {field.description && (
                    <Typography.Paragraph
                      type="secondary"
                      className="!text-xs !mb-1.5"
                    >
                      {field.description}
                    </Typography.Paragraph>
                  )}
                  <Input.Password
                    placeholder={
                      isSaved
                        ? "Already saved — enter a new value to replace"
                        : `Enter ${field.display_name || field.field_key}`
                    }
                    value={values[field.field_key] ?? ""}
                    onChange={(e) =>
                      setValues((prev) => ({
                        ...prev,
                        [field.field_key]: e.target.value,
                      }))
                    }
                    autoComplete="off"
                  />
                </div>
              );
            })}
          </div>
        )}

        <div className="mt-5 bg-blue-50 rounded-lg p-3 flex items-start gap-2 text-xs text-blue-700">
          <LockOutlined className="mt-0.5 flex-shrink-0" />
          <span>
            Your values are encrypted with the proxy&apos;s salt key and never
            logged. Other users on this proxy cannot see them.
          </span>
        </div>

        <div className="mt-4 flex gap-2 justify-end">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={loading || fetching}
            className="px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white rounded-md transition-colors"
          >
            {loading ? "Saving…" : "Save & Connect"}
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default UserFieldsModal;
