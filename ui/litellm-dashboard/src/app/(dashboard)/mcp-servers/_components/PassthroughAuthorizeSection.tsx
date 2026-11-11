import React from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Input } from "@/components/ui/input";
import { MultiSelect } from "@/components/shared/MultiSelect";
import DcrBridgeToggle from "./DcrBridgeToggle";
import { MountedFormField } from "@/components/common_components/MountedFormField";
import { tagsControl, textControl } from "./mcpFieldRules";
import { credentialAuthClass, isClientForwardedTokenMode } from "@/components/mcp_tools/types";

interface PassthroughOAuthFlow {
  startOAuthFlow: () => void | Promise<void>;
  status: string;
  error: string | null;
  tokenResponse: { access_token?: string; expires_in?: number } | null;
}

/**
 * Browser-only Authorize & Fetch for the client-forwarded token modes
 * (true_passthrough / oauth_delegate). Tokens are never stored: the token
 * obtained here lives in this browser session only, forwarded per-server for
 * the tools preview and allowlist configuration, and is never written to the
 * server row or the per-user credential store. The optional OAuth client
 * credentials cover IdPs without dynamic client registration (e.g. a
 * pre-registered Slack app); unlike the token they ARE saved onto the server
 * as declared config, so internal users' Authorize relays through the org's
 * app instead of dead-ending on upstreams that cannot mint clients.
 *
 * Blank fields follow the same convention as the M2M credential fields. On
 * create they mean "no app configured" (dynamic client registration). On edit
 * they mean "keep existing" ONLY when the credential class is unchanged: the
 * backend merges a partial update within the client-forwarded class, so a
 * true_passthrough <-> oauth_delegate switch keeps the stored app, but a switch
 * from a different class (e.g. oauth2) replaces it, so blanks then mean "no
 * app". Removing a stored app is an explicit checkbox (edit only) that writes
 * an explicit-null credential.
 */
export default function PassthroughAuthorizeSection({
  authType,
  oauthFlow,
  dcrBridgeInitialChecked,
  isEditing = false,
  savedAuthType,
  removeStoredApp = false,
  onRemoveStoredAppChange,
  appMayNotMatchUpstream = false,
}: {
  authType?: string | null;
  oauthFlow: PassthroughOAuthFlow;
  dcrBridgeInitialChecked?: boolean;
  isEditing?: boolean;
  savedAuthType?: string | null;
  removeStoredApp?: boolean;
  onRemoveStoredAppChange?: (remove: boolean) => void;
  appMayNotMatchUpstream?: boolean;
}) {
  const { t } = useTranslation();
  if (!isClientForwardedTokenMode(authType)) return null;
  const authorizeButtonLabels: Record<string, string> = {
    authorizing: t("mcpServers.passthroughAuthorizeSection.authorize.authorizing"),
    exchanging: t("mcpServers.passthroughAuthorizeSection.authorize.exchanging"),
  };
  const authorizeButtonLabel =
    authorizeButtonLabels[oauthFlow.status] ?? t("mcpServers.passthroughAuthorizeSection.authorize.start");
  // On edit, "keep existing" only holds when the stored credential class is unchanged; a cross-class
  // switch (e.g. oauth2 -> true_passthrough) replaces credentials, so blanks then mean "no app".
  const classUnchanged = isEditing && credentialAuthClass(savedAuthType) === credentialAuthClass(authType);
  const clientIdPlaceholder = classUnchanged
    ? t("mcpServers.passthroughAuthorizeSection.clientId.placeholderKeepSaved")
    : t("mcpServers.passthroughAuthorizeSection.clientId.placeholderDcr");
  const clientSecretPlaceholder = classUnchanged
    ? t("mcpServers.passthroughAuthorizeSection.clientSecret.placeholderKeepSaved")
    : t("mcpServers.passthroughAuthorizeSection.clientSecret.placeholderPublicClient");
  const fieldClassName = "rounded-lg border-border focus:border-info focus:ring-ring";
  const clientIdExtra = classUnchanged
    ? t("mcpServers.passthroughAuthorizeSection.clientId.helpKeepSaved")
    : t("mcpServers.passthroughAuthorizeSection.clientId.helpClassChanged");
  return (
    <div className="rounded-lg border border-dashed border-border p-4 space-y-2 mb-4">
      <p className="text-sm text-muted-foreground">{t("mcpServers.passthroughAuthorizeSection.description")}</p>
      {appMayNotMatchUpstream && (
        <p className="text-sm text-warning">{t("mcpServers.passthroughAuthorizeSection.appMayNotMatchUpstream")}</p>
      )}
      <MountedFormField
        label={
          <span className="text-sm font-medium text-foreground">
            {t("mcpServers.passthroughAuthorizeSection.clientId.label")}
          </span>
        }
        name={["credentials", "client_id"]}
        help={clientIdExtra}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={clientIdPlaceholder}
            disabled={removeStoredApp}
            groupClassName="rounded-lg border-border focus:border-info focus:ring-ring"
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <span className="text-sm font-medium text-foreground">
            {t("mcpServers.passthroughAuthorizeSection.clientSecret.label")}
          </span>
        }
        name={["credentials", "client_secret"]}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={clientSecretPlaceholder}
            disabled={removeStoredApp}
            groupClassName="rounded-lg border-border focus:border-info focus:ring-ring"
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <span className="text-sm font-medium text-foreground">
            {t("mcpServers.passthroughAuthorizeSection.authorizationUrl.label")}
          </span>
        }
        name="authorization_url"
        help={t("mcpServers.passthroughAuthorizeSection.authorizationUrl.help")}
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="https://login.microsoftonline.com/<tenant-id>/oauth2/v2.0/authorize"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <span className="text-sm font-medium text-foreground">
            {t("mcpServers.passthroughAuthorizeSection.tokenUrl.label")}
          </span>
        }
        name="token_url"
        help={t("mcpServers.passthroughAuthorizeSection.tokenUrl.help")}
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="https://login.microsoftonline.com/<tenant-id>/oauth2/v2.0/token"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <span className="text-sm font-medium text-foreground">
            {t("mcpServers.passthroughAuthorizeSection.scopes.label")}
          </span>
        }
        name={["credentials", "scopes"]}
        help={t("mcpServers.passthroughAuthorizeSection.scopes.help")}
      >
        {(control) => (
          <MultiSelect
            {...tagsControl(control, t)}
            placeholder={t("mcpServers.passthroughAuthorizeSection.scopes.placeholder")}
            className="rounded-lg"
          />
        )}
      </MountedFormField>
      <DcrBridgeToggle authType={authType} initialChecked={dcrBridgeInitialChecked} />
      {isEditing && onRemoveStoredAppChange && (
        <Label className="items-start leading-normal font-normal text-foreground">
          <Checkbox className="mt-0.5" checked={removeStoredApp} onCheckedChange={onRemoveStoredAppChange} />
          {t("mcpServers.passthroughAuthorizeSection.removeStoredApp")}
        </Label>
      )}
      <Button
        variant="outline"
        onClick={oauthFlow.startOAuthFlow}
        disabled={oauthFlow.status === "authorizing" || oauthFlow.status === "exchanging"}
      >
        {authorizeButtonLabel}
      </Button>
      {oauthFlow.error && <p className="text-sm text-destructive">{oauthFlow.error}</p>}
      {oauthFlow.status === "success" && oauthFlow.tokenResponse?.access_token && (
        <p className="text-sm text-success">{t("mcpServers.passthroughAuthorizeSection.authorize.success")}</p>
      )}
    </div>
  );
}
