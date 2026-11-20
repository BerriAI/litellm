import React from "react";
import { useTranslation } from "react-i18next";
import { KeyRound } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { AUTH_TYPE } from "@/components/mcp_tools/types";
import { getProxyBaseUrl } from "@/components/networking";

export default function SignedIdentityHint({
  authType,
  serverUrl,
}: {
  authType?: string | null;
  serverUrl?: string | null;
}) {
  const { t } = useTranslation();
  if (authType !== AUTH_TYPE.SIGNED_IDENTITY) return null;
  const rows: readonly (readonly [string, string])[] = [
    [t("mcpServers.signedIdentityHint.publicKeys"), `${getProxyBaseUrl()}/.well-known/jwks.json`],
    [t("mcpServers.signedIdentityHint.audience"), serverUrl || t("mcpServers.signedIdentityHint.audienceUnset")],
    [t("mcpServers.signedIdentityHint.userClaim"), t("mcpServers.signedIdentityHint.userClaimValue")],
  ];
  return (
    <Alert className="mb-4">
      <KeyRound />
      <AlertTitle>{t("mcpServers.signedIdentityHint.title")}</AlertTitle>
      <AlertDescription>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
          {rows.map(([label, value]) => (
            <React.Fragment key={label}>
              <dt className="font-medium">{label}</dt>
              <dd className="break-all font-mono text-xs">{value}</dd>
            </React.Fragment>
          ))}
        </dl>
        <p className="mt-2">{t("mcpServers.signedIdentityHint.signingKey")}</p>
      </AlertDescription>
    </Alert>
  );
}
