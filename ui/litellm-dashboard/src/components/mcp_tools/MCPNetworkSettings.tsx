import React, { useState, useEffect } from "react";
import { Select, Button, Card, Typography, Spin, Tag } from "antd";
import { SaveOutlined, PlusOutlined } from "@ant-design/icons";
import { getGeneralSettingsCall, updateConfigFieldSetting, deleteConfigFieldSetting, fetchMCPClientIp } from "../networking";

const { Text } = Typography;

interface MCPNetworkSettingsProps {
  accessToken: string | null;
}

/**
 * Given an IP like "203.0.113.45", return "203.0.113.0/24".
 */
function ipToSlash24(ip: string): string {
  const parts = ip.split(".");
  if (parts.length !== 4) return ip + "/32";
  return `${parts[0]}.${parts[1]}.${parts[2]}.0/24`;
}

const MCPNetworkSettings: React.FC<MCPNetworkSettingsProps> = ({ accessToken }) => {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [privateRanges, setPrivateRanges] = useState<string[]>([]);
  const [currentIp, setCurrentIp] = useState<string | null>(null);

  useEffect(() => {
    loadSettings();
    detectCurrentIp();
  }, [accessToken]);

  const loadSettings = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const settings = await getGeneralSettingsCall(accessToken);
      for (const field of settings) {
        if (field.field_name === "mcp_internal_ip_ranges" && field.field_value) {
          setPrivateRanges(field.field_value);
        }
      }
    } catch (error) {
      console.error("Failed to load MCP network settings:", error);
    } finally {
      setLoading(false);
    }
  };

  const detectCurrentIp = async () => {
    if (!accessToken) return;
    const ip = await fetchMCPClientIp(accessToken);
    if (ip) {
      setCurrentIp(ip);
    }
  };

  const handleSave = async () => {
    if (!accessToken) return;
    setSaving(true);
    try {
      if (privateRanges.length > 0) {
        await updateConfigFieldSetting(accessToken, "mcp_internal_ip_ranges", privateRanges);
      } else {
        await deleteConfigFieldSetting(accessToken, "mcp_internal_ip_ranges");
      }
    } catch (error) {
      console.error("Failed to save MCP network settings:", error);
    } finally {
      setSaving(false);
    }
  };

  const addSuggestedRange = (range: string) => {
    if (!privateRanges.includes(range)) {
      setPrivateRanges([...privateRanges, range]);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <Spin />
      </div>
    );
  }

  const suggestedRange = currentIp ? ipToSlash24(currentIp) : null;

  return (
    <div className="space-y-6 p-4">
      <div>
        <Text className="text-lg font-semibold">Private IP Ranges</Text>
        <p className="text-sm text-gray-500 mt-1">
          Define which IP ranges are part of your private network. Callers from these IPs can see all MCP servers. Callers from any other IP can only see servers marked &quot;Available on Public Internet&quot;.
        </p>
      </div>

      <Card>
        {currentIp && (
          <div className="mb-4 p-3 bg-blue-50 rounded-lg">
            <Text className="text-sm text-blue-700">
              Your current IP: <span className="font-mono font-medium">{currentIp}</span>
            </Text>
            {suggestedRange && !privateRanges.includes(suggestedRange) && (
              <div className="mt-1">
                <Text className="text-sm text-blue-600">Suggested range: </Text>
                <Tag
                  className="cursor-pointer font-mono"
                  color="blue"
                  icon={<PlusOutlined />}
                  onClick={() => addSuggestedRange(suggestedRange)}
                >
                  {suggestedRange}
                </Tag>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center mb-2">
          <Text className="font-medium">Your Private Network Ranges</Text>
        </div>
        <Select
          mode="tags"
          value={privateRanges}
          onChange={setPrivateRanges}
          placeholder="Leave empty to use defaults: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8"
          tokenSeparators={[","]}
          className="w-full"
          size="large"
          allowClear
        />
        <p className="text-xs text-gray-400 mt-2">
          Enter CIDR ranges (e.g., 10.0.0.0/8). When empty, standard private IP ranges are used.
        </p>
      </Card>

      <div className="flex justify-end">
        <Button
          type="primary"
          icon={<SaveOutlined />}
          onClick={handleSave}
          loading={saving}
        >
          Save
        </Button>
      </div>
    </div>
  );
};

export default MCPNetworkSettings;
