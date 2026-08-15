import React, { useState, useEffect } from "react";
import { Save, Plus, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import {
  getGeneralSettingsCall,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
  fetchMCPClientIp,
} from "@/components/networking";

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
  const [rangeDraft, setRangeDraft] = useState("");

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

  // Commas separate entries, matching the old tokenised input.
  const commitDraft = () => {
    const added = rangeDraft
      .split(",")
      .map((r) => r.trim())
      .filter((r) => r !== "" && !privateRanges.includes(r));
    if (added.length > 0) {
      setPrivateRanges([...privateRanges, ...added]);
    }
    setRangeDraft("");
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <UiLoadingSpinner className="size-6 text-muted-foreground" />
      </div>
    );
  }

  const suggestedRange = currentIp ? ipToSlash24(currentIp) : null;

  return (
    <div className="space-y-6 p-4">
      <DeprecationBanner featureName="MCP Network Settings and the internal-network-only flag" />
      <div>
        <p className="text-lg font-semibold">Private IP Ranges</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Define which IP ranges are part of your private network. Callers from these IPs can see all MCP servers.
          Callers from any other IP can only see servers marked &quot;Available on Public Internet&quot;.
        </p>
      </div>

      <Card className="p-6">
        {currentIp && (
          <div className="mb-4 rounded-lg bg-muted p-3">
            <p className="text-sm">
              Your current IP: <span className="font-mono font-medium">{currentIp}</span>
            </p>
            {suggestedRange && !privateRanges.includes(suggestedRange) && (
              <div className="mt-1 flex items-center gap-2">
                <p className="text-sm">Suggested range: </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="font-mono"
                  onClick={() => addSuggestedRange(suggestedRange)}
                >
                  <Plus />
                  {suggestedRange}
                </Button>
              </div>
            )}
          </div>
        )}

        <div className="mb-2 flex items-center">
          <p className="text-sm font-medium">Your Private Network Ranges</p>
        </div>
        {privateRanges.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {privateRanges.map((range) => (
              <Badge key={range} variant="secondary" className="font-mono">
                {range}
                <button
                  type="button"
                  aria-label={`Remove ${range}`}
                  onClick={() => setPrivateRanges(privateRanges.filter((r) => r !== range))}
                  className="ml-1 cursor-pointer"
                >
                  <X className="size-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}
        <Input
          value={rangeDraft}
          placeholder="Leave empty to use defaults: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8"
          onChange={(e) => setRangeDraft(e.target.value)}
          onBlur={commitDraft}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              commitDraft();
            }
          }}
        />
        <p className="mt-2 text-xs text-muted-foreground">
          Enter CIDR ranges (e.g., 10.0.0.0/8). When empty, standard private IP ranges are used.
        </p>
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={saving}>
          <Save />
          Save
        </Button>
      </div>
    </div>
  );
};

export default MCPNetworkSettings;
