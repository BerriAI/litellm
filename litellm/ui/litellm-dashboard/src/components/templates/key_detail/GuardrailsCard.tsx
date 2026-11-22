"use client";

import { Shield } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/shared/table_cells";

import { BadgeRow, SectionCard, SubHead, ValueSkeleton } from "./SectionCard";

interface GuardrailsCardProps {
  guardrails: readonly string[];
  policies: readonly string[];
  policyGuardrails: Readonly<Record<string, readonly string[]>>;
  loadingPolicies: boolean;
  globalGuardrailsOff: boolean;
}

function GuardrailBadge({ name }: { name: string }) {
  return (
    <Badge variant="outline" className="gap-1">
      <Shield className="size-3" aria-hidden />
      {name}
    </Badge>
  );
}

function PolicyAdds({ loading, added }: { loading: boolean; added: readonly string[] }) {
  const { t } = useTranslation();
  if (loading) return <ValueSkeleton width={80} height={20} />;
  if (added.length === 0)
    return <span className="text-muted-foreground">{t("keyDetail.overview.guardrails.addsNone")}</span>;
  return (
    <>
      <span className="text-muted-foreground">{t("keyDetail.overview.guardrails.adds")}</span>
      <BadgeRow>
        {added.map((guardrail) => (
          <GuardrailBadge key={guardrail} name={guardrail} />
        ))}
      </BadgeRow>
    </>
  );
}

export function GuardrailsCard({
  guardrails,
  policies,
  policyGuardrails,
  loadingPolicies,
  globalGuardrailsOff,
}: GuardrailsCardProps) {
  const { t } = useTranslation();
  return (
    <SectionCard icon={Shield} title={t("keyDetail.overview.guardrails.title")} testId="key-guardrails-card">
      {guardrails.length > 0 ? (
        <div className="flex flex-col gap-2">
          <SubHead title={t("keyDetail.overview.guardrails.onThisKey")} count={guardrails.length} />
          <BadgeRow>
            {guardrails.map((guardrail) => (
              <GuardrailBadge key={guardrail} name={guardrail} />
            ))}
          </BadgeRow>
        </div>
      ) : null}
      {policies.length > 0 ? (
        <div className="flex flex-col gap-2">
          <SubHead title={t("keyDetail.overview.guardrails.fromPolicies")} count={policies.length} />
          {policies.map((policy) => {
            const added = policyGuardrails[policy] ?? [];
            return (
              <div key={policy} className="flex flex-wrap items-center gap-1.5">
                <Badge variant="secondary">{policy}</Badge>
                <PolicyAdds loading={loadingPolicies} added={added} />
              </div>
            );
          })}
        </div>
      ) : null}
      <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3 text-muted-foreground">
        {globalGuardrailsOff ? (
          <>
            <StatusBadge tone="warning" label={t("keyDetail.overview.guardrails.globalOff")} />
            {t("keyDetail.overview.guardrails.globalOffNote")}
          </>
        ) : (
          t("keyDetail.overview.guardrails.globalOn")
        )}
      </div>
    </SectionCard>
  );
}
