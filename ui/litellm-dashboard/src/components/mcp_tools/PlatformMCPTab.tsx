import React, { useEffect, useState } from "react";
import { ExperimentOutlined, SaveOutlined, ToolOutlined } from "@ant-design/icons";
import { Button, Card, InputNumber, Spin, Switch, Typography } from "antd";
import { getConfigFieldSetting, updateConfigFieldSetting } from "../networking";

const { Text } = Typography;

const PLATFORM_MCP_ENABLED_FIELD = "platform_mcp_enabled";
const PLATFORM_MCP_THRESHOLD_FIELD = "platform_mcp_tool_threshold";
const DEFAULT_THRESHOLD = 10;

interface PlatformMCPTabProps {
  accessToken: string | null;
}

const getFieldValue = (response: unknown, fallback: boolean | number): boolean | number => {
  if (response && typeof response === "object") {
    const field = response as { field_value?: unknown; field_default_value?: unknown };
    if (field.field_value !== undefined && field.field_value !== null) {
      return field.field_value as boolean | number;
    }
    if (field.field_default_value !== undefined && field.field_default_value !== null) {
      return field.field_default_value as boolean | number;
    }
  }
  return fallback;
};

const PlatformMCPTab: React.FC<PlatformMCPTabProps> = ({ accessToken }) => {
  const [loading, setLoading] = useState(true);
  const [savingEnabled, setSavingEnabled] = useState(false);
  const [savingThreshold, setSavingThreshold] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);

  useEffect(() => {
    const loadSettings = async () => {
      if (!accessToken) {
        setLoading(false);
        return;
      }

      setLoading(true);
      try {
        const [enabledResponse, thresholdResponse] = await Promise.all([
          getConfigFieldSetting(accessToken, PLATFORM_MCP_ENABLED_FIELD),
          getConfigFieldSetting(accessToken, PLATFORM_MCP_THRESHOLD_FIELD),
        ]);
        setEnabled(Boolean(getFieldValue(enabledResponse, false)));
        const nextThreshold = Number(getFieldValue(thresholdResponse, DEFAULT_THRESHOLD));
        setThreshold(Number.isFinite(nextThreshold) && nextThreshold > 0 ? nextThreshold : DEFAULT_THRESHOLD);
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

  const handleSaveThreshold = async () => {
    if (!accessToken) return;
    setSavingThreshold(true);
    try {
      await updateConfigFieldSetting(accessToken, PLATFORM_MCP_THRESHOLD_FIELD, threshold);
    } catch (error) {
      console.error("Failed to update Platform MCP threshold:", error);
    } finally {
      setSavingThreshold(false);
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
          When enabled, LiteLLM compresses aggregate MCP tools/list responses only after the caller&apos;s filtered tool
          count is over the configured threshold.
        </p>
      </div>

      <Card>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Text className="font-medium">Enable Platform MCP compression</Text>
            <p className="mb-0 mt-1 text-sm text-gray-500">
              Disabled returns the current full tool list. Enabled keeps the full tool list at or below the threshold,
              then returns only list_servers and enable_server above it.
            </p>
          </div>
          <Switch checked={enabled} loading={savingEnabled} onChange={handleToggle} />
        </div>
      </Card>

      <Card>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Text className="font-medium">Compression threshold</Text>
            <p className="mb-0 mt-1 text-sm text-gray-500">
              Default is 10 tools. Compression starts when the final accessible tool count is greater than this value.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <InputNumber min={1} value={threshold} onChange={(value) => setThreshold(Number(value || 1))} />
            <Button type="primary" icon={<SaveOutlined />} loading={savingThreshold} onClick={handleSaveThreshold}>
              Save
            </Button>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
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
              <Text className="font-mono font-semibold text-blue-600">enable_server</Text>
              <p className="mb-0 mt-1 text-sm text-gray-500">
                Returns full tool definitions for one accessible MCP server.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default PlatformMCPTab;
