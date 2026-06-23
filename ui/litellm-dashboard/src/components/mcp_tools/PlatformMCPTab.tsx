import React, { useEffect, useState } from "react";
import { ExperimentOutlined, ToolOutlined } from "@ant-design/icons";
import { Card, Spin, Switch, Typography } from "antd";
import { getConfigFieldSetting, updateConfigFieldSetting } from "../networking";

const { Text } = Typography;

const PLATFORM_MCP_ENABLED_FIELD = "platform_mcp_enabled";

interface PlatformMCPTabProps {
  accessToken: string | null;
}

const getFieldValue = (response: unknown, fallback: boolean): boolean => {
  if (response && typeof response === "object") {
    const field = response as { field_value?: unknown; field_default_value?: unknown };
    if (field.field_value !== undefined && field.field_value !== null) {
      return Boolean(field.field_value);
    }
    if (field.field_default_value !== undefined && field.field_default_value !== null) {
      return Boolean(field.field_default_value);
    }
  }
  return fallback;
};

const PlatformMCPTab: React.FC<PlatformMCPTabProps> = ({ accessToken }) => {
  const [loading, setLoading] = useState(true);
  const [savingEnabled, setSavingEnabled] = useState(false);
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    const loadSettings = async () => {
      if (!accessToken) {
        setLoading(false);
        return;
      }

      setLoading(true);
      try {
        const enabledResponse = await getConfigFieldSetting(accessToken, PLATFORM_MCP_ENABLED_FIELD);
        setEnabled(getFieldValue(enabledResponse, false));
      } catch (error) {
        console.error("Failed to load Platform MCP settings:", error);
      } finally {
        setLoading(false);
      }
    };

    loadSettings();
  }, [accessToken]);

  const handleToggle = async (checked: boolean) => {
    if (!accessToken) return;
    setSavingEnabled(true);
    try {
      await updateConfigFieldSetting(accessToken, PLATFORM_MCP_ENABLED_FIELD, checked);
      setEnabled(checked);
    } catch (error) {
      console.error("Failed to update Platform MCP enabled setting:", error);
    } finally {
      setSavingEnabled(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <Spin />
      </div>
    );
  }

  return (
    <div className="space-y-6 p-4">
      <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
        <ExperimentOutlined className="flex-shrink-0 text-amber-600" />
        <span>
          Platform MCP is a pre-v0 feature. This can change unexpectedly. Do not use this in production. If you have
          feedback, please email product@berri.ai.
        </span>
      </div>

      <div>
        <Text className="text-lg font-semibold">Platform MCP</Text>
        <p className="mt-1 text-sm text-gray-500">
          When enabled, LiteLLM exposes platform-managed MCP discovery and tool calling through the proxy.
        </p>
      </div>

      <Card>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Text className="font-medium">Enable Platform MCP</Text>
            <p className="mb-0 mt-1 text-sm text-gray-500">
              Disabled leaves existing MCP behavior unchanged. Enabled makes list_servers, get_server_tools, and
              call_tool available for keys with platform_mcp access.
            </p>
          </div>
          <Switch aria-label="Enable Platform MCP" checked={enabled} loading={savingEnabled} onChange={handleToggle} />
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card>
          <div className="flex items-start gap-3">
            <ToolOutlined className="mt-1 text-gray-500" />
            <div>
              <Text className="font-mono font-semibold text-blue-600">list_servers</Text>
              <p className="mb-0 mt-1 text-sm text-gray-500">
                Returns MCP server names and descriptions for servers the key can access.
              </p>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-start gap-3">
            <ToolOutlined className="mt-1 text-gray-500" />
            <div>
              <Text className="font-mono font-semibold text-blue-600">get_server_tools</Text>
              <p className="mb-0 mt-1 text-sm text-gray-500">
                Returns full tool definitions for one accessible MCP server.
              </p>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-start gap-3">
            <ToolOutlined className="mt-1 text-gray-500" />
            <div>
              <Text className="font-mono font-semibold text-blue-600">call_tool</Text>
              <p className="mb-0 mt-1 text-sm text-gray-500">
                Calls a tool on an accessible MCP server through the platform.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default PlatformMCPTab;
