import { type FC, useState } from "react";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { cn } from "@/lib/cva.config";
import { toast } from "@/lib/toast";
import { deleteMCPOAuthUserCredential, deleteMCPServerOAuthToken } from "@/components/networking";

export type MCPDisconnectScope = "server" | "self";

export type MCPDisconnectMode = "disconnect" | "reauthorize";

export const MCP_DISCONNECT_SCOPE_COPY: Record<MCPDisconnectScope, { label: string; blastRadius: string }> = {
  server: {
    label: "Every stored token for this server (affects all users)",
    blastRadius:
      "Clears every OAuth token LiteLLM holds for this server, for every user. Anyone who authorized interactively loses upstream access until they authorize again, and their OAuth app stays configured. A machine-to-machine server also loses the stored client credentials it mints from, so an admin has to re-enter the client secret before it can call upstream again. Stored BYOK API keys are left alone.",
  },
  self: {
    label: "Only your own connection (affects just you)",
    blastRadius:
      "Clears only the token stored against your user. Every other user keeps their connection and the server stays authorized for them.",
  },
};

interface MCPDisconnectDialogProps {
  open: boolean;
  mode: MCPDisconnectMode;
  serverId: string;
  serverName?: string;
  accessToken: string;
  isProxyAdmin: boolean;
  onOpenChange: (open: boolean) => void;
  onCleared: (scope: MCPDisconnectScope) => void;
}

const MCPDisconnectDialog: FC<MCPDisconnectDialogProps> = ({
  open,
  mode,
  serverId,
  serverName,
  accessToken,
  isProxyAdmin,
  onOpenChange,
  onCleared,
}) => {
  const [scope, setScope] = useState<MCPDisconnectScope>(isProxyAdmin ? "server" : "self");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const confirmLabel = mode === "reauthorize" ? "Clear & Reauthorize" : "Disconnect";

  const handleConfirm = async () => {
    setIsSubmitting(true);
    try {
      if (scope === "server") {
        const result = await deleteMCPServerOAuthToken(accessToken, serverId);
        toast.success(
          result.cleared
            ? `Cleared every stored OAuth token for this server (${result.cleared_user_tokens} user token${
                result.cleared_user_tokens === 1 ? "" : "s"
              })${
                result.cleared_client_credentials
                  ? ". Re-enter the client credentials to let this server mint tokens again"
                  : ""
              }`
            : "This server had no stored OAuth token",
        );
      } else {
        await deleteMCPOAuthUserCredential(accessToken, serverId);
        toast.success("Cleared your connection to this MCP server");
      }
      onOpenChange(false);
      onCleared(scope);
    } catch (error) {
      toast.fromError(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            {mode === "reauthorize" ? "Reauthorize MCP Server?" : "Disconnect MCP Server?"}
          </AlertDialogTitle>
        </AlertDialogHeader>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            {mode === "reauthorize"
              ? "Clearing the stored token lets you run Authorize & Fetch Token again and grant a different set of upstream scopes. The server's URL, auth type and issuer are kept, so nothing has to be recreated."
              : "Choose what to clear. The server's URL, auth type and issuer are kept, so the server can be authorized again without being recreated."}
          </p>
          <dl className="space-y-1 rounded-lg border border-border bg-muted p-4">
            {serverName && (
              <div className="flex gap-2">
                <dt className="text-sm text-muted-foreground">Name</dt>
                <dd className="text-sm font-semibold">{serverName}</dd>
              </div>
            )}
            <div className="flex gap-2">
              <dt className="text-sm text-muted-foreground">ID</dt>
              <dd className="font-mono text-xs">{serverId}</dd>
            </div>
          </dl>
          <RadioGroup value={scope} onValueChange={(value) => setScope(value as MCPDisconnectScope)}>
            {(["server", "self"] as const).map((option) => {
              const disabled = option === "server" && !isProxyAdmin;
              return (
                <label
                  key={option}
                  onClick={() => !disabled && setScope(option)}
                  className={cn(
                    "flex cursor-pointer items-start gap-3 rounded-lg border p-3",
                    scope === option ? "border-primary bg-accent/40" : "border-border",
                    disabled && "cursor-not-allowed opacity-60",
                  )}
                >
                  <RadioGroupItem value={option} disabled={disabled} className="mt-1" />
                  <div className="space-y-1">
                    <Label className="cursor-pointer font-semibold">{MCP_DISCONNECT_SCOPE_COPY[option].label}</Label>
                    <p className="text-xs text-muted-foreground">{MCP_DISCONNECT_SCOPE_COPY[option].blastRadius}</p>
                    {disabled && <p className="text-xs text-destructive">Requires proxy admin.</p>}
                  </div>
                </label>
              );
            })}
          </RadioGroup>
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isSubmitting}>Cancel</AlertDialogCancel>
          <Button variant="destructive" disabled={isSubmitting} onClick={handleConfirm}>
            {isSubmitting ? "Clearing..." : confirmLabel}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};

export default MCPDisconnectDialog;
