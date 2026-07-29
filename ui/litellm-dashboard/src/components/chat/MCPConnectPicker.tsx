import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import MessageManager from "@/components/molecules/message_manager";
import { Logo } from "@/components/molecules/logo/Logo";
import { fetchMCPServers, listMCPTools } from "../networking";
import { MCPServer } from "../mcp_tools/types";

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const MCPConnectPicker: React.FC<Props> = ({ accessToken, selectedServers, onChange }) => {
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
        const list: MCPServer[] = Array.isArray(data) ? data : data?.data ?? [];
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
        MessageManager.warning(`Could not load tools for ${serverName} \u2014 it will be excluded from this message.`);
        return;
      }
      onChange([...selectedServers, serverName]);
    } catch {
      MessageManager.warning(`Could not load tools for ${serverName} \u2014 it will be excluded from this message.`);
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
        <div className="flex flex-col gap-1">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="flex items-center justify-between px-3 py-2 gap-3">
              <div className="flex items-start gap-3 flex-1 min-w-0">
                <Skeleton className="h-6 w-6 rounded-md shrink-0" />
                <div className="flex flex-col gap-1.5 flex-1 min-w-0">
                  <Skeleton className="h-3.5 w-24" />
                  <Skeleton className="h-3 w-32" />
                </div>
              </div>
              <Skeleton className="h-3.5 w-6 rounded-full shrink-0" />
            </div>
          ))}
        </div>
      ) : servers.length === 0 ? (
        <div className="px-3 py-4 text-muted-foreground text-[13px] text-center">No MCP servers configured</div>
      ) : (
        servers.map((server) => {
          const name = server.server_name ?? server.alias ?? server.server_id;
          const isSelected = selectedServers.includes(name);
          const isTogglingOn = togglingOn.has(name);

          return (
            <div key={server.server_id} className="flex items-start justify-between px-3 py-2 gap-3">
              {server.mcp_info?.logo_url && (
                <Logo
                  src={server.mcp_info.logo_url}
                  label={name}
                  className="w-6 h-6 rounded-md object-contain shrink-0 mt-0.5"
                />
              )}
              <div className="flex-1 min-w-0">
                <div className="font-medium text-[13px] text-foreground truncate">{name}</div>
                {server.description && (
                  <div className="text-xs text-muted-foreground mt-0.5 truncate">{server.description}</div>
                )}
              </div>
              <div className="relative shrink-0">
                {isTogglingOn ? (
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                ) : (
                  <Switch
                    checked={isSelected}
                    onCheckedChange={(checked) => handleToggle(name, checked)}
                    className="scale-75"
                  />
                )}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
};

export default MCPConnectPicker;
