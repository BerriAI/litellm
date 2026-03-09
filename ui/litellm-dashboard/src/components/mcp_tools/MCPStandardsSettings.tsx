"use client";

import React, { useEffect, useState, useCallback } from "react";
import { getGeneralSettingsCall, updateConfigFieldSetting } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import { MCPServer } from "./types";

interface MCPStandardsSettingsProps {
  accessToken: string | null;
}

export interface RequiredFieldDef {
  key: string;
  label: string;
  description: string;
  check: (server: MCPServer) => boolean;
}

export const MCP_REQUIRED_FIELD_DEFS: RequiredFieldDef[] = [
  {
    key: "description",
    label: "Description",
    description: "Server must have a non-empty description.",
    check: (s) => !!s.description?.trim(),
  },
  {
    key: "source_url",
    label: "GitHub / Source URL",
    description: "Server must have a link to the source repository.",
    check: (s) => !!s.source_url?.trim(),
  },
  {
    key: "alias",
    label: "Alias",
    description: "Server must have a human-readable alias.",
    check: (s) => !!s.alias?.trim(),
  },
  {
    key: "auth_type",
    label: "Auth configured",
    description: "Server must have an auth type set (not 'none').",
    check: (s) => !!s.auth_type && s.auth_type !== "none",
  },
  {
    key: "url",
    label: "Server URL",
    description: "Server must have a URL configured.",
    check: (s) => !!s.url?.trim(),
  },
];

const SETTINGS_KEY = "mcp_required_fields";

export default function MCPStandardsSettings({ accessToken }: MCPStandardsSettingsProps) {
  const [requiredFields, setRequiredFields] = useState<string[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const loadSettings = useCallback(async () => {
    if (!accessToken) return;
    setIsLoading(true);
    try {
      const settings = await getGeneralSettingsCall(accessToken);
      const rows: Array<{ field_name: string; field_value: unknown }> = Array.isArray(settings?.data)
        ? settings.data
        : [];
      const row = rows.find((r) => r.field_name === SETTINGS_KEY);
      if (row && Array.isArray(row.field_value)) {
        setRequiredFields(row.field_value as string[]);
      }
    } catch {
      // leave defaults
    } finally {
      setIsLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const toggleField = (key: string) => {
    setRequiredFields((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  const handleSave = async () => {
    if (!accessToken) return;
    setIsSaving(true);
    try {
      await updateConfigFieldSetting(accessToken, SETTINGS_KEY, requiredFields);
      NotificationsManager.success("Standards saved");
    } catch {
      NotificationsManager.fromBackend("Failed to save standards");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="p-6 max-w-2xl">
      <div className="mb-6">
        <h2 className="text-base font-semibold text-gray-900">MCP Submission Standards</h2>
        <p className="text-sm text-gray-500 mt-1">
          Choose which fields are required for a submission to pass your standards. Each submission
          card in the Team MCPs tab will show a green ✓ or red ✗ for each requirement.
        </p>
      </div>

      {isLoading ? (
        <div className="text-sm text-gray-400">Loading…</div>
      ) : (
        <div className="space-y-3">
          {MCP_REQUIRED_FIELD_DEFS.map((field) => {
            const enabled = requiredFields.includes(field.key);
            return (
              <div
                key={field.key}
                className={`flex items-center justify-between px-4 py-3 rounded-lg border transition-colors ${
                  enabled ? "border-blue-200 bg-blue-50" : "border-gray-200 bg-white"
                }`}
              >
                <div className="flex-1 min-w-0 mr-4">
                  <div className="text-sm font-medium text-gray-900">{field.label}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{field.description}</div>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={enabled}
                  onClick={() => toggleField(field.key)}
                  className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none ${
                    enabled ? "bg-blue-500" : "bg-gray-200"
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                      enabled ? "translate-x-4" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>
            );
          })}
        </div>
      )}

      <div className="mt-6">
        <button
          type="button"
          disabled={isSaving || isLoading}
          onClick={handleSave}
          className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-md transition-colors"
        >
          {isSaving ? "Saving…" : "Save Standards"}
        </button>
      </div>
    </div>
  );
}
