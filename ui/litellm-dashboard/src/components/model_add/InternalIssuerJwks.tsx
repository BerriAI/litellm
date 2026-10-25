import { useLanguage } from "@/contexts/LanguageContext";
import CopyButton from "@/components/shared/CopyButton";
import { FieldDescription, FieldLegend, FieldSet } from "@/components/ui/field";
import { $api } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import type { JwksPanel } from "./credential_federation";

const SavedJwks = ({ credentialName }: { credentialName: string }) => {
  const { t } = useLanguage();
  const jwks = $api.useQuery("get", "/credentials/{credential_name}/jwks", {
    params: { path: { credential_name: credentialName } },
  });
  if (jwks.isPending || !jwks.isFetchedAfterMount) {
    return <p className="text-sm text-muted-foreground">{t("Loading JWKS...")}</p>;
  }
  if (jwks.isError) {
    return (
      <p role="alert" className="text-sm text-destructive">
        {extractProxyErrorMessage(jwks.error)}
      </p>
    );
  }
  const jwksText = JSON.stringify(jwks.data, null, 2);
  return (
    <div className="flex flex-col gap-2">
      <pre aria-label={t("Public JWKS")} className="max-h-48 overflow-auto rounded-md border bg-muted p-3 text-xs">
        {jwksText}
      </pre>
      <CopyButton value={jwksText} label={t("Copy JWKS")} variant="action" className="self-start" />
    </div>
  );
};

export default function InternalIssuerJwks({ panel }: { panel: JwksPanel }) {
  const { t } = useLanguage();
  switch (panel.kind) {
    case "hidden":
      return null;
    case "after_save":
      return (
        <p className="mb-4 text-sm text-muted-foreground">
          {t(
            "Once saved, reopen this credential from LLM Credentials to copy the public JWKS you register in the Claude Console.",
          )}
        </p>
      );
    case "saved":
      return (
        <FieldSet className="mb-4 gap-2">
          <FieldLegend variant="label" className="mb-0">
            {t("Public JWKS")}
          </FieldLegend>
          <FieldDescription>
            {t(
              "Register this JWKS with the Issuer URL and Subject above as a federation rule under Settings > Workload identity in the Claude Console. Then fill in its Federation Rule ID and Organization ID at the top of this form and update the credential. The JWKS comes from the saved signing key, so update after changing the key and before copying.",
            )}
          </FieldDescription>
          <SavedJwks credentialName={panel.credentialName} />
        </FieldSet>
      );
  }
}
