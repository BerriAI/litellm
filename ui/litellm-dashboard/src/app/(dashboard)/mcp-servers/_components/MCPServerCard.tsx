import { useState, type FC, type KeyboardEvent, type MouseEvent } from "react";
import { Dropdown, Tooltip, Typography, Tag } from "antd";
import type { MenuProps } from "antd";
import {
  CheckOutlined,
  DeleteOutlined,
  ExclamationCircleFilled,
  MoreOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { AUTH_TYPE, type MCPServer } from "@/components/mcp_tools/types";
import { getMaskedAndFullUrl } from "./utils";

const { Text } = Typography;

interface MCPServerCardProps {
  server: MCPServer;
  // Per-user env-var fields this user still needs to fill in for this server.
  // Computed by the parent from the bulk /user-env-vars/status response, so
  // the card never issues a per-row request (no N+1).
  missingUserFields?: string[];
  isLoadingHealth?: boolean;
  isRechecking?: boolean;
  onClick: () => void;
  onRecheckHealth?: () => void;
  onByokConnect?: () => void;
  onOpenFillFields?: () => void;
  onDelete?: () => void;
}

const HEALTH_TONE: Record<string, { dot: string }> = {
  healthy: { dot: "bg-green-500" },
  unhealthy: { dot: "bg-red-500" },
  unknown: { dot: "bg-gray-300" },
};

// Stop card-level click handler from firing when an interactive child is used.
const stop = (e: MouseEvent | KeyboardEvent) => e.stopPropagation();

const MCPServerCard: FC<MCPServerCardProps> = ({
  server,
  missingUserFields,
  isLoadingHealth,
  isRechecking,
  onClick,
  onRecheckHealth,
  onByokConnect,
  onOpenFillFields,
  onDelete,
}) => {
  const alias = server.alias || server.server_name || "";
  const name = server.server_name || alias || server.server_id;
  // Logo is sourced exclusively from the admin-set `mcp_info.logo_url`.
  const candidateLogo = server.mcp_info?.logo_url ?? undefined;
  const [failedLogoUrl, setFailedLogoUrl] = useState<string | null>(null);
  const logoUrl = candidateLogo && failedLogoUrl !== candidateLogo ? candidateLogo : undefined;
  const transport = server.transport || "http";
  const displayTransport = server.spec_path && transport !== "stdio" ? "openapi" : transport;
  const authType = server.auth_type || "none";
  // An oauth2 server with no persisted oauth2_flow was never classified as M2M vs
  // interactive; flag it so an admin can set it from the edit page (see the OAuth
  // Flow Type selector) instead of leaving LiteLLM to fall back to a default.
  // Delegate (PKCE passthrough) servers authenticate upstream and route to
  // passthrough regardless of oauth2_flow, so the classification does not apply to
  // them and they are not flagged.
  const oauthFlowUnset =
    server.auth_type === AUTH_TYPE.OAUTH2 && !server.oauth2_flow && !server.delegate_auth_to_upstream;
  const status = server.status || "unknown";
  const healthTone = HEALTH_TONE[status] ?? HEALTH_TONE.unknown;
  const isPublic = server.available_on_public_internet;
  const accessGroups = (server.mcp_access_groups ?? []).filter((g): g is string => typeof g === "string");

  const missing = missingUserFields ?? [];
  const needsAttention = missing.length > 0;

  const cardClass = needsAttention
    ? "border-2 border-red-300 bg-red-50/40 hover:border-red-400 hover:shadow-md"
    : "border border-gray-200 bg-white hover:border-gray-300 hover:shadow-md";

  const url = server.url || "";
  const { maskedUrl } = url ? getMaskedAndFullUrl(url) : { maskedUrl: "" };

  // Transport-adapted identifier shown under the title. Every transport has
  // something useful here, which keeps the tag row vertically aligned across
  // cards in the grid (stdio cards no longer "snap up" because they lack a URL).
  let subtitle = "";
  let subtitleTooltip = "";
  if (transport === "stdio") {
    const parts = [server.command, ...(server.args ?? [])].filter(
      (p): p is string => typeof p === "string" && p.length > 0,
    );
    subtitle = parts.join(" ");
    subtitleTooltip = subtitle;
  } else if (server.spec_path) {
    subtitle = server.spec_path;
    subtitleTooltip = server.spec_path;
  } else if (url) {
    subtitle = maskedUrl;
    subtitleTooltip = url;
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onClick();
    }
  };

  const menuItems: MenuProps["items"] = [];
  if (onRecheckHealth) {
    menuItems.push({
      key: "test-connection",
      label: "Test Connection",
      icon: <ThunderboltOutlined />,
      disabled: isRechecking,
      onClick: ({ domEvent }) => {
        domEvent.stopPropagation();
        onRecheckHealth();
      },
    });
  }
  if (onDelete) {
    if (menuItems.length > 0) {
      menuItems.push({ key: "divider", type: "divider" });
    }
    menuItems.push({
      key: "delete",
      label: "Delete",
      icon: <DeleteOutlined />,
      danger: true,
      onClick: ({ domEvent }) => {
        domEvent.stopPropagation();
        onDelete();
      },
    });
  }

  // Card uses role="button" + nested <button> children (Set, BYOK Connect, the
  // recheck-health Tag), so a real <button> wrapper would produce invalid
  // nested-interactive HTML. The role + tabIndex + Enter/Space handler keeps
  // the whole card clickable and keyboard-accessible.
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      className={`group relative flex h-full cursor-pointer flex-col gap-3 rounded-lg p-4 transition-all duration-150 focus:outline-hidden focus-visible:ring-2 focus-visible:ring-blue-400 ${cardClass}`}
    >
      <div className="flex items-start gap-3">
        {logoUrl ? (
          <img
            src={logoUrl}
            alt={`${name} logo`}
            className="h-10 w-10 shrink-0 rounded-sm object-contain"
            onError={() => setFailedLogoUrl(logoUrl)}
          />
        ) : (
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-sm bg-gray-100 font-semibold text-gray-500">
            {(name || "?").slice(0, 2).toUpperCase()}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="block w-full truncate text-left font-semibold text-gray-900" title={name}>
            {name}
          </div>
          <div className="mt-0.5 flex items-center gap-2 text-xs text-gray-500">
            {alias && <span className="truncate">{alias}</span>}
            {alias && <span className="text-gray-300">·</span>}
            <Tooltip title={server.server_id}>
              <span className="font-mono text-blue-600">{server.server_id.slice(0, 7)}</span>
            </Tooltip>
          </div>
        </div>
        {menuItems.length > 0 && (
          <Dropdown menu={{ items: menuItems }} trigger={["click"]} placement="bottomRight">
            <button
              type="button"
              onClick={stop}
              onKeyDown={stop}
              aria-label="Server actions"
              className="-mr-1 -mt-1 inline-flex h-8 w-8 items-center justify-center rounded-md text-gray-500 transition-colors hover:bg-gray-100 hover:text-blue-600"
            >
              <MoreOutlined style={{ fontSize: 20 }} />
            </button>
          </Dropdown>
        )}
      </div>

      {subtitle ? (
        <Tooltip title={subtitleTooltip}>
          <Text className="truncate font-mono text-xs text-gray-500" ellipsis>
            {subtitle}
          </Text>
        </Tooltip>
      ) : (
        // Defensive placeholder: keep the row even when no identifier is
        // available so the tag row stays vertically aligned across the grid.
        <div className="h-[18px]" aria-hidden />
      )}

      <div className="flex flex-wrap items-center gap-1.5">
        <HealthChip
          status={status}
          isLoadingHealth={isLoadingHealth}
          isRechecking={isRechecking}
          onRecheck={onRecheckHealth}
          lastCheck={server.last_health_check}
          error={server.health_check_error}
          dotClass={healthTone.dot}
        />
        <Tag className="m-0">{displayTransport.toUpperCase()}</Tag>
        <Tag className="m-0">{authType}</Tag>
        {oauthFlowUnset && (
          <Tooltip title="This OAuth server has no flow set (Machine-to-Machine vs Interactive). Open it and choose an OAuth Flow Type so LiteLLM authenticates it as you intend.">
            <Tag color="warning" className="m-0">
              <span className="inline-flex items-center gap-1">
                <ExclamationCircleFilled />
                OAuth flow not set
              </span>
            </Tag>
          </Tooltip>
        )}
        <Tag color={isPublic ? "green" : "orange"} className="m-0">
          <span className="inline-flex items-center gap-1">
            <span className={`h-1.5 w-1.5 rounded-full ${isPublic ? "bg-green-500" : "bg-orange-500"}`} />
            {isPublic ? "Public" : "Internal"}
          </span>
        </Tag>
        {accessGroups.slice(0, 2).map((g) => (
          <Tooltip key={g} title={g}>
            <Tag className="m-0 max-w-[120px] truncate">{g}</Tag>
          </Tooltip>
        ))}
        {accessGroups.length > 2 && (
          <Tooltip title={accessGroups.slice(2).join(", ")}>
            <Tag className="m-0">+{accessGroups.length - 2}</Tag>
          </Tooltip>
        )}
      </div>

      {(server.is_byok || needsAttention) && (
        <div className="mt-auto flex flex-col gap-2">
          {server.is_byok && <ByokRow connected={!!server.has_user_credential} onConnect={onByokConnect} />}
          {needsAttention && (
            <div className="flex items-center justify-between gap-2 text-xs">
              <Tooltip
                title={
                  <div>
                    <div className="font-semibold mb-1">Missing user fields:</div>
                    <ul className="ml-3">
                      {missing.map((m) => (
                        <li key={m}>• {m}</li>
                      ))}
                    </ul>
                  </div>
                }
              >
                <span className="inline-flex items-center gap-1 font-semibold text-red-700">
                  <ExclamationCircleFilled />
                  {missing.length} user field
                  {missing.length === 1 ? "" : "s"} missing
                </span>
              </Tooltip>
              {onOpenFillFields && (
                <button
                  type="button"
                  onClick={(e) => {
                    stop(e);
                    onOpenFillFields();
                  }}
                  className="rounded-md bg-red-600 px-3 py-1 text-xs font-medium text-white shadow-xs transition-colors hover:bg-red-700"
                >
                  Set
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

interface HealthChipProps {
  status: string;
  isLoadingHealth?: boolean;
  isRechecking?: boolean;
  onRecheck?: () => void;
  lastCheck?: string | null;
  error?: string | null;
  dotClass: string;
}

const HealthChip: FC<HealthChipProps> = ({
  status,
  isLoadingHealth,
  isRechecking,
  onRecheck,
  lastCheck,
  error,
  dotClass,
}) => {
  if (isLoadingHealth || isRechecking) {
    return (
      <Tag className="m-0">
        <span className="inline-flex items-center gap-1.5 text-xs text-gray-500">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-300" />
          Checking
        </span>
      </Tag>
    );
  }
  const tooltip = (
    <div className="max-w-xs">
      <div className="font-semibold mb-1">Health: {status}</div>
      {lastCheck && <div className="text-xs mb-1">Last check: {new Date(lastCheck).toLocaleString()}</div>}
      {error && (
        <div className="text-xs">
          <div className="font-medium text-red-300 mb-1">Error</div>
          <div className="wrap-break-word">{error}</div>
        </div>
      )}
      {!lastCheck && !error && <div className="text-xs text-gray-400">No health data</div>}
      {onRecheck && <div className="mt-1 text-xs text-gray-300">Click to recheck</div>}
    </div>
  );
  return (
    <Tooltip title={tooltip} placement="top">
      <Tag
        className={`m-0 ${onRecheck ? "cursor-pointer hover:opacity-80" : "cursor-default"}`}
        onClick={
          onRecheck
            ? (e) => {
                e.stopPropagation();
                onRecheck();
              }
            : undefined
        }
      >
        <span className="inline-flex items-center gap-1.5">
          <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </Tag>
    </Tooltip>
  );
};

interface ByokRowProps {
  connected: boolean;
  onConnect?: () => void;
}

const ByokRow: FC<ByokRowProps> = ({ connected, onConnect }) => {
  if (connected) {
    return (
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="text-gray-500">BYOK credential</span>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full border border-green-200 bg-green-50 px-2 py-0.5 font-medium text-green-700">
            <CheckOutlined style={{ fontSize: 10 }} /> Connected
          </span>
          {onConnect && (
            <button
              type="button"
              onClick={(e) => {
                stop(e);
                onConnect();
              }}
              className="text-xs text-gray-400 transition-colors hover:text-blue-600"
            >
              Update
            </button>
          )}
        </div>
      </div>
    );
  }
  return (
    <div className="flex items-center justify-between gap-2 text-xs">
      <span className="text-gray-500">BYOK credential</span>
      {onConnect ? (
        <button
          type="button"
          onClick={(e) => {
            stop(e);
            onConnect();
          }}
          className="rounded-md bg-blue-600 px-3 py-1 text-xs font-medium text-white shadow-xs transition-colors hover:bg-blue-700"
        >
          Connect
        </button>
      ) : (
        <span className="text-gray-400">—</span>
      )}
    </div>
  );
};

export default MCPServerCard;
