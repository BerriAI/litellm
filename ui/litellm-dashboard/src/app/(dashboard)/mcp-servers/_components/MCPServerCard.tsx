import { type FC, type KeyboardEvent, type MouseEvent } from "react";
import { Check, CircleAlert, Ellipsis, Trash2, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { AUTH_TYPE, type MCPServer } from "@/components/mcp_tools/types";
import { Logo } from "@/components/molecules/logo/Logo";
import { getMaskedAndFullUrl } from "./utils";

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
    ? "border-2 border-destructive/40 bg-destructive/5 hover:border-destructive/60 hover:shadow-md"
    : "border border-border bg-card hover:shadow-md";

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

  const hasMenu = !!onRecheckHealth || !!onDelete;

  // Card uses role="button" + nested <button> children (Set, BYOK Connect, the
  // recheck-health Badge), so a real <button> wrapper would produce invalid
  // nested-interactive HTML. The role + tabIndex + Enter/Space handler keeps
  // the whole card clickable and keyboard-accessible.
  return (
    <TooltipProvider>
      <div
        role="button"
        tabIndex={0}
        onClick={onClick}
        onKeyDown={handleKeyDown}
        className={cn(
          "group relative flex h-full cursor-pointer flex-col gap-3 rounded-lg p-4 transition-all duration-150 focus:outline-hidden focus-visible:ring-2 focus-visible:ring-ring",
          cardClass,
        )}
      >
        <div className="flex items-start gap-3">
          {candidateLogo ? (
            <Logo src={candidateLogo} label={name} className="h-10 w-10 shrink-0 rounded-sm object-contain" />
          ) : (
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-sm bg-muted font-semibold text-muted-foreground">
              {(name || "?").slice(0, 2).toUpperCase()}
            </div>
          )}
          <div className="min-w-0 flex-1">
            <div className="block w-full truncate text-left font-semibold" title={name}>
              {name}
            </div>
            <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
              {alias && <span className="truncate">{alias}</span>}
              {alias && <span>·</span>}
              <Tooltip>
                <TooltipTrigger
                  render={<span className="font-mono text-primary">{server.server_id.slice(0, 7)}</span>}
                />
                <TooltipContent>{server.server_id}</TooltipContent>
              </Tooltip>
            </div>
          </div>
          {hasMenu && (
            <DropdownMenu>
              <DropdownMenuTrigger
                render={
                  <button
                    type="button"
                    onClick={stop}
                    onKeyDown={stop}
                    aria-label="Server actions"
                    className="-mr-1 -mt-1 inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                  >
                    <Ellipsis className="size-5" />
                  </button>
                }
              />
              <DropdownMenuContent align="end">
                {onRecheckHealth && (
                  <DropdownMenuItem
                    disabled={isRechecking}
                    onClick={(e) => {
                      stop(e);
                      onRecheckHealth();
                    }}
                  >
                    <Zap />
                    Test Connection
                  </DropdownMenuItem>
                )}
                {onRecheckHealth && onDelete && <DropdownMenuSeparator />}
                {onDelete && (
                  <DropdownMenuItem
                    variant="destructive"
                    onClick={(e) => {
                      stop(e);
                      onDelete();
                    }}
                  >
                    <Trash2 />
                    Delete
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>

        {subtitle ? (
          <Tooltip>
            <TooltipTrigger render={<p className="truncate font-mono text-xs text-muted-foreground">{subtitle}</p>} />
            <TooltipContent>{subtitleTooltip}</TooltipContent>
          </Tooltip>
        ) : (
          // Defensive placeholder: keep the row even when no identifier is
          // available so the badge row stays vertically aligned across the grid.
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
          <Badge variant="outline">{displayTransport.toUpperCase()}</Badge>
          <Badge variant="outline">{authType}</Badge>
          {oauthFlowUnset && (
            <Tooltip>
              <TooltipTrigger
                render={
                  <Badge variant="outline">
                    <CircleAlert />
                    OAuth flow not set
                  </Badge>
                }
              />
              <TooltipContent>
                This OAuth server has no flow set (Machine-to-Machine vs Interactive). Open it and choose an OAuth Flow
                Type so LiteLLM authenticates it as you intend.
              </TooltipContent>
            </Tooltip>
          )}
          <Badge variant="outline">
            <span className={cn("h-1.5 w-1.5 rounded-full", isPublic ? "bg-green-500" : "bg-orange-500")} />
            {isPublic ? "Public" : "Internal"}
          </Badge>
          {accessGroups.slice(0, 2).map((g) => (
            <Tooltip key={g}>
              <TooltipTrigger
                render={
                  <Badge variant="outline" className="max-w-[120px] truncate">
                    {g}
                  </Badge>
                }
              />
              <TooltipContent>{g}</TooltipContent>
            </Tooltip>
          ))}
          {accessGroups.length > 2 && (
            <Tooltip>
              <TooltipTrigger render={<Badge variant="outline">+{accessGroups.length - 2}</Badge>} />
              <TooltipContent>{accessGroups.slice(2).join(", ")}</TooltipContent>
            </Tooltip>
          )}
        </div>

        {(server.is_byok || needsAttention) && (
          <div className="mt-auto flex flex-col gap-2">
            {server.is_byok && <ByokRow connected={!!server.has_user_credential} onConnect={onByokConnect} />}
            {needsAttention && (
              <div className="flex items-center justify-between gap-2 text-xs">
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <span className="inline-flex items-center gap-1 font-semibold text-destructive">
                        <CircleAlert className="size-3.5" />
                        {missing.length} user field
                        {missing.length === 1 ? "" : "s"} missing
                      </span>
                    }
                  />
                  <TooltipContent>
                    <div className="mb-1 font-semibold">Missing user fields:</div>
                    <ul className="ml-3">
                      {missing.map((m) => (
                        <li key={m}>• {m}</li>
                      ))}
                    </ul>
                  </TooltipContent>
                </Tooltip>
                {onOpenFillFields && (
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={(e) => {
                      stop(e);
                      onOpenFillFields();
                    }}
                  >
                    Set
                  </Button>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </TooltipProvider>
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
      <Badge variant="outline" className="text-muted-foreground">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted-foreground" />
        Checking
      </Badge>
    );
  }
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Badge
            variant="outline"
            className={onRecheck ? "cursor-pointer hover:opacity-80" : "cursor-default"}
            onClick={
              onRecheck
                ? (e) => {
                    e.stopPropagation();
                    onRecheck();
                  }
                : undefined
            }
          >
            <span className={cn("h-1.5 w-1.5 rounded-full", dotClass)} />
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </Badge>
        }
      />
      <TooltipContent side="top" className="max-w-xs">
        <div className="mb-1 font-semibold">Health: {status}</div>
        {lastCheck && <div className="mb-1 text-xs">Last check: {new Date(lastCheck).toLocaleString()}</div>}
        {error && (
          <div className="text-xs">
            <div className="mb-1 font-medium">Error</div>
            <div className="wrap-break-word">{error}</div>
          </div>
        )}
        {!lastCheck && !error && <div className="text-xs">No health data</div>}
        {onRecheck && <div className="mt-1 text-xs">Click to recheck</div>}
      </TooltipContent>
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
        <span className="text-muted-foreground">BYOK credential</span>
        <div className="flex items-center gap-2">
          <Badge variant="outline">
            <Check /> Connected
          </Badge>
          {onConnect && (
            <Button
              variant="link"
              size="sm"
              onClick={(e) => {
                stop(e);
                onConnect();
              }}
            >
              Update
            </Button>
          )}
        </div>
      </div>
    );
  }
  return (
    <div className="flex items-center justify-between gap-2 text-xs">
      <span className="text-muted-foreground">BYOK credential</span>
      {onConnect ? (
        <Button
          size="sm"
          onClick={(e) => {
            stop(e);
            onConnect();
          }}
        >
          Connect
        </Button>
      ) : (
        <span className="text-muted-foreground">—</span>
      )}
    </div>
  );
};

export default MCPServerCard;
