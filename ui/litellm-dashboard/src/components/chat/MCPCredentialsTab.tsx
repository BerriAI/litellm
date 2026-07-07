"use client";

import React, { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Trash2, Link } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import MessageManager from "@/components/molecules/message_manager";
import { deleteMCPOAuthUserCredential, listMCPUserCredentials, MCPUserCredentialListItem } from "../networking";

const MCP_CREDENTIALS_QUERY_KEY = "mcp-user-credentials";

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

function expiryLabel(isoString: string | null | undefined): {
  text: string;
  variant: "secondary" | "destructive" | "outline";
} {
  if (!isoString) return { text: "Does not expire", variant: "secondary" };
  try {
    const exp = new Date(isoString);
    const diffMs = exp.getTime() - Date.now();
    if (diffMs <= 0) return { text: "Expired", variant: "destructive" };
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHr / 24);
    if (diffDay > 0) return { text: `Expires in ${diffDay}d`, variant: "outline" };
    if (diffHr > 0) return { text: `Expires in ${diffHr}h`, variant: "outline" };
    return { text: `Expires in ${diffMin}m`, variant: "outline" };
  } catch {
    return { text: "", variant: "outline" };
  }
}

const MCPCredentialsTab: React.FC<Props> = ({ accessToken }) => {
  const queryClient = useQueryClient();
  const [revoking, setRevoking] = useState<Set<string>>(new Set());

  const { data: credentials = [], isLoading: loading } = useQuery({
    queryKey: [MCP_CREDENTIALS_QUERY_KEY, accessToken],
    queryFn: () => listMCPUserCredentials(accessToken),
    enabled: !!accessToken,
  });

  const handleRevoke = async (serverId: string) => {
    setRevoking((prev) => new Set(prev).add(serverId));
    try {
      await deleteMCPOAuthUserCredential(accessToken, serverId);
      queryClient.setQueryData<MCPUserCredentialListItem[]>([MCP_CREDENTIALS_QUERY_KEY, accessToken], (prev) =>
        (prev ?? []).filter((c) => c.server_id !== serverId),
      );
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

  const displayName = (c: MCPUserCredentialListItem) => c.alias || c.server_name || c.server_id;

  return (
    <div className="w-full">
      <div className="mb-4">
        <h2 className="text-base font-semibold text-foreground mb-0.5">App Credentials</h2>
        <p className="text-sm text-muted-foreground m-0">Your stored OAuth connections; used automatically in chat</p>
      </div>

      {loading ? (
        <div className="rounded-lg border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  App
                </TableHead>
                <TableHead className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Connected
                </TableHead>
                <TableHead className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Status
                </TableHead>
                <TableHead className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground text-right">
                  Actions
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Array.from({ length: 3 }, (_, i) => (
                <TableRow key={i}>
                  <TableCell>
                    <Skeleton className="h-4 w-24" />
                  </TableCell>
                  <TableCell>
                    <Skeleton className="h-4 w-16" />
                  </TableCell>
                  <TableCell>
                    <Skeleton className="h-4 w-20" />
                  </TableCell>
                  <TableCell className="text-right">
                    <Skeleton className="h-4 w-8 ml-auto" />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      ) : credentials.length === 0 ? (
        <div className="text-center text-muted-foreground text-sm py-12 border border-dashed rounded-lg">
          <Link className="h-6 w-6 mb-3 mx-auto text-muted-foreground/50" />
          <p className="m-0">No connections yet</p>
          <p className="m-0 mt-1 text-xs">
            Go to <span className="font-medium">Integrations</span> and click{" "}
            <span className="font-medium">Connect</span> to authorize an MCP server
          </p>
        </div>
      ) : (
        <div className="rounded-lg border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  App
                </TableHead>
                <TableHead className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Connected
                </TableHead>
                <TableHead className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Status
                </TableHead>
                <TableHead className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground text-right">
                  Actions
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {credentials.map((cred) => {
                const isRevoking = revoking.has(cred.server_id);
                const exp = expiryLabel(cred.expires_at);
                return (
                  <TableRow key={cred.server_id}>
                    <TableCell className="text-sm font-medium">{displayName(cred)}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {relativeTime(cred.connected_at) || "\u2014"}
                    </TableCell>
                    <TableCell>
                      <Badge variant={exp.variant}>{exp.text}</Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            variant="outline"
                            size="icon-sm"
                            disabled={isRevoking}
                            title="Revoke connection"
                            className="text-muted-foreground hover:text-destructive hover:border-destructive/50"
                          >
                            {isRevoking ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Trash2 className="h-3.5 w-3.5" />
                            )}
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Revoke connection?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This removes the stored OAuth credential for {displayName(cred)}. You&apos;ll need to
                              reconnect to use it in chat again.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction variant="destructive" onClick={() => handleRevoke(cred.server_id)}>
                              Revoke
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
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
