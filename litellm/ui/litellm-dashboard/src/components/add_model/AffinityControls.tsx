import React from "react";
import { useTranslation } from "react-i18next";

import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { DEFAULT_DEPLOYMENT_AFFINITY, DEFAULT_SESSION_AFFINITY_TTL_SECONDS } from "./ComplexityRouterConfig";

export const AffinityControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const [ttlDraft, setTtlDraft] = React.useState<string | null>(null);
  const commitTtl = (raw: string) => {
    setTtlDraft(null);
    if (raw.trim() === "") {
      onChange({ ...value, session_affinity_ttl_seconds: undefined });
      return;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    onChange({ ...value, session_affinity_ttl_seconds: Math.max(1, Math.round(parsed)) });
  };

  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={value.deployment_affinity ?? DEFAULT_DEPLOYMENT_AFFINITY}
          onCheckedChange={(deploymentAffinity) => onChange({ ...value, deployment_affinity: deploymentAffinity })}
          aria-label={t("addModel.affinityControls.pinDeployment")}
        />
        <strong className="font-semibold">{t("addModel.affinityControls.pinDeployment")}</strong>
      </div>
      <span className="block text-xs mb-3 text-muted-foreground">
        {t("addModel.affinityControls.pinDeploymentDescription")}
      </span>
      <div style={{ maxWidth: 320 }}>
        <label className="block text-sm font-medium mb-1" htmlFor="session-affinity-ttl">
          {t("addModel.affinityControls.pinTtl")}
        </label>
        <Input
          id="session-affinity-ttl"
          inputMode="numeric"
          value={ttlDraft ?? value.session_affinity_ttl_seconds ?? ""}
          placeholder={String(DEFAULT_SESSION_AFFINITY_TTL_SECONDS)}
          onChange={(event) => setTtlDraft(event.target.value)}
          onBlur={(event) => commitTtl(event.target.value)}
        />
        <span className="block text-xs mt-1 text-muted-foreground">
          {t("addModel.affinityControls.pinTtlHelp", { seconds: DEFAULT_SESSION_AFFINITY_TTL_SECONDS })}
        </span>
      </div>
    </>
  );
};
