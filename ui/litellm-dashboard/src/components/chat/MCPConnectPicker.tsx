import React, { useEffect, useState } from "react";
import { Switch } from "@/components/ui/switch";
import { Loader2 } from "lucide-react";
import MessageManager from "@/components/molecules/message_manager";
import { fetchMCPServers, listMCPTools } from "../networking";
import { MCPServer } from "../mcp_tools/types";

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const MCPConnectPicker: React.FC<Props> = ({
  accessToken,
  selectedServers,
  onChange,
}) => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loadingServers, setLoadingServers] = useState(true);
  const [togglingOn, setTogglingOn] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoadingServers(true);
      try {
        const data = await fetchMCPServers(accessToken);
        if (cancelled) return;
        const list: MCPServer[] = Array.isArray(data)
          ? data
          : (data?.data ?? []);
        setServers(list);
      } catch {
        if (!cancelled) {
          setServers([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingServers(false);
        }
      }
    };

    load();

    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  const handleToggle = async (serverName: string, checked: boolean) => {
    if (!checked) {
      onChange(selectedServers.filter((s) => s !== serverName));
      return;
    }

    setTogglingOn((prev) => new Set(prev).add(serverName));
    try {
      const result = await listMCPTools(accessToken, serverName);
      if (result?.error) {
        MessageManager.warning(
          `Could not load tools for ${serverName} — it will be excluded from this message.`,
        );
        return;
      }
      onChange([...selectedServers, serverName]);
    } catch {
      MessageManager.warning(
        `Could not load tools for ${serverName} — it will be excluded from this message.`,
      );
    } finally {
      setTogglingOn((prev) => {
        const next = new Set(prev);
        next.delete(serverName);
        return next;
      });
    }
  };

  return (
    <div className="max-w-[320px] max-h-[400px] overflow-y-auto py-2">
      {loadingServers ? (
        <div className="flex justify-center py-6">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        </div>
      ) : servers.length === 0 ? (
        <div className="px-3 py-4 text-muted-foreground text-sm text-center">
          No MCP servers configured
        </div>
      ) : (
        servers.map((server) => {
          const name = server.server_name ?? server.alias ?? server.server_id;
          const isSelected = selectedServers.includes(name);
          const isTogglingOn = togglingOn.has(name);

          return (
            <div
              key={server.server_id}
              className="flex items-start justify-between px-3 py-2 gap-3"
            >
              {server.mcp_info?.logo_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={server.mcp_info.logo_url}
                  alt={`${name} logo`}
                  className="w-6 h-6 rounded-md object-contain shrink-0 mt-0.5"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              )}
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm text-foreground truncate">
                  {name}
                </div>
                {server.description && (
                  <div className="text-xs text-muted-foreground mt-0.5 truncate">
                    {server.description}
                  </div>
                )}
              </div>
              <div className="flex items-center">
                {isTogglingOn && (
                  <Loader2 className="h-3 w-3 animate-spin text-muted-foreground mr-1" />
                )}
                <Switch
                  checked={isSelected}
                  onCheckedChange={(checked) => handleToggle(name, checked)}
                  disabled={isTogglingOn}
                />
              </div>
            </div>
          );
        })
      )}
    </div>
  );
};

export default MCPConnectPicker;
