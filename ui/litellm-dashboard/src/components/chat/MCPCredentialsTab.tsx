"use client";

/**
 * MCPCredentialsTab
 *
 * Shows all OAuth2 MCP connections the calling user has stored.
 * Lives in the Chat sidebar's "Credentials" tab.
 */

import React, { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { Link as LinkIcon, Loader2, Trash2 } from "lucide-react";
import MessageManager from "@/components/molecules/message_manager";
import {
  deleteMCPOAuthUserCredential,
  listMCPUserCredentials,
  MCPUserCredentialListItem,
} from "../networking";

interface Props {
  accessToken: string;
}

function relativeTime(isoString: string | null | undefined): string {
  if (!isoString) return "";
  try {
    const date = new Date(isoString);
    const diffMs = Date.now() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 60) return "just now";
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    return `${Math.floor(diffHr / 24)}d ago`;
  } catch {
    return "";
  }
}

function expiryLabel(isoString: string | null | undefined): string {
  if (!isoString) return "Does not expire";
  try {
    const exp = new Date(isoString);
    const diffMs = exp.getTime() - Date.now();
    if (diffMs <= 0) return "Expired";
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHr / 24);
    if (diffDay > 0) return `Expires in ${diffDay}d`;
    if (diffHr > 0) return `Expires in ${diffHr}h`;
    return `Expires in ${diffMin}m`;
  } catch {
    return "";
  }
}

const MCPCredentialsTab: React.FC<Props> = ({ accessToken }) => {
  const [credentials, setCredentials] = useState<MCPUserCredentialListItem[]>(
    [],
  );
  const [loading, setLoading] = useState(true);
  const [revoking, setRevoking] = useState<Set<string>>(new Set());

  const load = useCallback(() => {
    setLoading(true);
    listMCPUserCredentials(accessToken)
      .then(setCredentials)
      .catch(() => setCredentials([]))
      .finally(() => setLoading(false));
  }, [accessToken]);

  useEffect(() => {
    load();
  }, [load]);

  const handleRevoke = async (serverId: string) => {
    setRevoking((prev) => new Set(prev).add(serverId));
    try {
      await deleteMCPOAuthUserCredential(accessToken, serverId);
      setCredentials((prev) => prev.filter((c) => c.server_id !== serverId));
    } catch {
      MessageManager.error("Failed to revoke connection. Please try again.");
    } finally {
      setRevoking((prev) => {
        const n = new Set(prev);
        n.delete(serverId);
        return n;
      });
    }
  };

  const displayName = (c: MCPUserCredentialListItem) =>
    c.alias || c.server_name || c.server_id;

  return (
    <div className="w-full">
      <div className="mb-4">
        <h2 className="text-base font-semibold text-foreground mb-0.5">
          App Credentials
        </h2>
        <p className="text-sm text-muted-foreground m-0">
          Your stored OAuth connections — used automatically in chat.
        </p>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : credentials.length === 0 ? (
        <div className="text-center text-muted-foreground text-sm py-12 border border-dashed border-border rounded-lg">
          <LinkIcon className="h-6 w-6 mx-auto mb-3 text-muted-foreground/60" />
          No connections yet.
          <br />
          Go to <strong>Apps</strong> and click <strong>Connect</strong> to
          authorize an MCP server.
        </div>
      ) : (
        <div className="rounded-lg border border-border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs font-medium">App</TableHead>
                <TableHead className="text-xs font-medium">
                  Connected
                </TableHead>
                <TableHead className="text-xs font-medium">Status</TableHead>
                <TableHead className="text-xs font-medium text-right">
                  Actions
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {credentials.map((cred) => {
                const name = displayName(cred);
                const isRevoking = revoking.has(cred.server_id);
                const exp = expiryLabel(cred.expires_at);
                const connected = relativeTime(cred.connected_at);
                const isExpired = exp === "Expired";

                return (
                  <TableRow
                    key={cred.server_id}
                    className="h-10 hover:bg-muted"
                  >
                    <TableCell>
                      <span className="text-sm font-medium text-foreground">
                        {name}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className="text-sm text-muted-foreground">
                        {connected || "—"}
                      </span>
                    </TableCell>
                    <TableCell>
                      <Badge
                        className={cn(
                          "text-xs",
                          isExpired
                            ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                            : "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
                        )}
                      >
                        {exp}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <button
                        type="button"
                        onClick={() => handleRevoke(cred.server_id)}
                        disabled={isRevoking}
                        title="Revoke connection"
                        className={cn(
                          "inline-flex items-center justify-center rounded-md border border-border px-2 py-1 text-muted-foreground hover:text-destructive hover:border-destructive/30 transition-colors bg-transparent",
                          isRevoking
                            ? "opacity-50 cursor-not-allowed"
                            : "cursor-pointer",
                        )}
                      >
                        {isRevoking ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Trash2 className="h-3.5 w-3.5" />
                        )}
                      </button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
};

export default MCPCredentialsTab;
