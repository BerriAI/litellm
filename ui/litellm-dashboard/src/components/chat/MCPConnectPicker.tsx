import React, { useEffect, useState } from "react";
import { Switch, Spin, message } from "antd";
import { LockOutlined } from "@ant-design/icons";
import { fetchMCPServers, listMCPTools } from "../networking";
import { MCPServer, AUTH_TYPE } from "../mcp_tools/types";

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

function isOAuthServer(server: MCPServer): boolean {
  return server.auth_type === AUTH_TYPE.OAUTH2 || server.auth_type === "oauth2";
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
        const list: MCPServer[] = Array.isArray(data) ? data : (data?.data ?? []);
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
        message.warning(
          `Could not load tools for ${serverName} — it will be excluded from this message.`
        );
        return;
      }
      onChange([...selectedServers, serverName]);
    } catch {
      message.warning(
        `Could not load tools for ${serverName} — it will be excluded from this message.`
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
    <div
      style={{
        maxWidth: 320,
        maxHeight: 400,
        overflowY: "auto",
        padding: "8px 0",
      }}
    >
      {loadingServers ? (
        <div style={{ display: "flex", justifyContent: "center", padding: "24px 0" }}>
          <Spin />
        </div>
      ) : servers.length === 0 ? (
        <div style={{ padding: "16px 12px", color: "#8c8c8c", fontSize: 13, textAlign: "center" }}>
          No MCP servers configured
        </div>
      ) : (
        servers.map((server) => {
          const name = server.server_name ?? server.alias ?? server.server_id;
          const isSelected = selectedServers.includes(name);
          const isTogglingOn = togglingOn.has(name);
          const isOAuth = isOAuthServer(server);

          return (
            <div
              key={server.server_id}
              style={{
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "space-between",
                padding: "8px 12px",
                gap: 12,
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    fontWeight: 500,
                    fontSize: 13,
                    color: "#1f1f1f",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    display: "flex",
                    alignItems: "center",
                    gap: 5,
                  }}
                >
                  {name}
                  {isOAuth && (
                    <LockOutlined style={{ fontSize: 10, color: "#d97706", flexShrink: 0 }} />
                  )}
                </div>
                {server.description && (
                  <div
                    style={{
                      fontSize: 12,
                      color: "#8c8c8c",
                      marginTop: 2,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {server.description}
                  </div>
                )}
              </div>
              <Switch
                size="small"
                checked={isSelected}
                loading={isTogglingOn}
                onChange={(checked) => handleToggle(name, checked)}
              />
            </div>
          );
        })
      )}
    </div>
  );
};

export default MCPConnectPicker;
