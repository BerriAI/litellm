import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Plus, Save, X } from "lucide-react";
import {
  getGeneralSettingsCall,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
  fetchMCPClientIp,
} from "../networking";

interface MCPNetworkSettingsProps {
  accessToken: string | null;
}

function ipToSlash24(ip: string): string {
  const parts = ip.split(".");
  if (parts.length !== 4) return ip + "/32";
  return `${parts[0]}.${parts[1]}.${parts[2]}.0/24`;
}

const MCPNetworkSettings: React.FC<MCPNetworkSettingsProps> = ({
  accessToken,
}) => {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [privateRanges, setPrivateRanges] = useState<string[]>([]);
  const [currentIp, setCurrentIp] = useState<string | null>(null);
  const [newRangeInput, setNewRangeInput] = useState("");

  useEffect(() => {
    const loadSettings = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        const settings = await getGeneralSettingsCall(accessToken);
        for (const field of settings) {
          if (
            field.field_name === "mcp_internal_ip_ranges" &&
            field.field_value
          ) {
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
      if (ip) setCurrentIp(ip);
    };
    loadSettings();
    detectCurrentIp();
  }, [accessToken]);

  const handleSave = async () => {
    if (!accessToken) return;
    setSaving(true);
    try {
      if (privateRanges.length > 0) {
        await updateConfigFieldSetting(
          accessToken,
          "mcp_internal_ip_ranges",
          privateRanges,
        );
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

  const commitNewRange = () => {
    const v = newRangeInput.trim();
    if (!v) return;
    if (!privateRanges.includes(v)) {
      setPrivateRanges([...privateRanges, v]);
    }
    setNewRangeInput("");
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <Skeleton className="h-8 w-48" />
      </div>
    );
  }

  const suggestedRange = currentIp ? ipToSlash24(currentIp) : null;

  return (
    <div className="space-y-6 p-4">
      <div>
        <span className="text-lg font-semibold">Private IP Ranges</span>
        <p className="text-sm text-muted-foreground mt-1">
          Define which IP ranges are part of your private network. Callers from
          these IPs can see all MCP servers. Callers from any other IP can only
          see servers marked &quot;Available on Public Internet&quot;.
        </p>
      </div>

      <Card className="p-4">
        {currentIp && (
          <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
            <span className="text-sm text-blue-700 dark:text-blue-300">
              Your current IP:{" "}
              <span className="font-mono font-medium">{currentIp}</span>
            </span>
            {suggestedRange && !privateRanges.includes(suggestedRange) && (
              <div className="mt-1 flex items-center gap-1">
                <span className="text-sm text-blue-600 dark:text-blue-400">
                  Suggested range:
                </span>
                <Badge
                  className="cursor-pointer font-mono gap-1"
                  onClick={() => addSuggestedRange(suggestedRange)}
                >
                  <Plus className="h-3 w-3" />
                  {suggestedRange}
                </Badge>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center mb-2">
          <span className="font-medium">Your Private Network Ranges</span>
        </div>
        <div className="space-y-2">
          <div className="flex gap-2">
            <Input
              value={newRangeInput}
              onChange={(e) => setNewRangeInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === ",") {
                  e.preventDefault();
                  commitNewRange();
                }
              }}
              placeholder="Leave empty to use defaults: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8"
              className="flex-1"
            />
            <Button
              type="button"
              variant="outline"
              onClick={commitNewRange}
              disabled={!newRangeInput.trim()}
            >
              <Plus className="h-4 w-4" />
              Add
            </Button>
          </div>
          {privateRanges.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {privateRanges.map((range) => (
                <Badge
                  key={range}
                  variant="secondary"
                  className="font-mono gap-1"
                >
                  {range}
                  <button
                    type="button"
                    onClick={() =>
                      setPrivateRanges(privateRanges.filter((r) => r !== range))
                    }
                    aria-label={`Remove ${range}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
            </div>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Enter CIDR ranges (e.g., 10.0.0.0/8). When empty, standard private IP
          ranges are used.
        </p>
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={saving}>
          <Save className="h-4 w-4" />
          {saving ? "Saving…" : "Save"}
        </Button>
      </div>
    </div>
  );
};

export default MCPNetworkSettings;
