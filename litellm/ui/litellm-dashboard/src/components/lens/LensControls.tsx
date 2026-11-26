"use client";

import Image from "next/image";
import Link from "next/link";
import { useId, useState } from "react";
import { ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";
import lensSymbol from "../../../public/assets/logos/lens-symbol.svg";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { TabsTrigger } from "@/components/ui/tabs";
import { useControlPlane } from "@/contexts/ControlPlaneContext";
import { $api } from "@/lib/http/api";
import { toast } from "@/lib/toast";
import { effectiveLensEnabled, type LensPolicy } from "./lensPolicy";

function LensSymbol({ size = 20 }: { size?: 20 | 28 }) {
  return <Image src={lensSymbol} alt="" width={size} height={size} loading="eager" unoptimized />;
}

export function LensTabTrigger() {
  const { t } = useTranslation();
  const tooltipId = useId();
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger
          aria-describedby={tooltipId}
          render={<TabsTrigger value="lens" className="flex-none rounded-none px-4 py-2" />}
        >
          <LensSymbol />
          {t("lensControls.title")}
          <Badge variant="secondary" className="h-4 rounded-sm px-1 text-xs text-muted-foreground">
            {t("lensControls.new")}
          </Badge>
        </TooltipTrigger>
        <TooltipContent id={tooltipId} role="tooltip" side="bottom">
          {t("lensControls.description")}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface LensControlsProps {
  kind: "team" | "key";
  policy: LensPolicy;
  teamPolicy?: LensPolicy;
  teamPolicyStatus?: "ready" | "loading" | "error";
  canEdit: boolean;
  href: string | null;
  onPolicyChange: (policy: LensPolicy) => Promise<void>;
}

function helperKey(kind: LensControlsProps["kind"], inherited: boolean) {
  if (kind === "team") return "lensControls.teamDefault";
  if (inherited) return "lensControls.keyInherited";
  return "lensControls.keyCustom";
}

type LensStatusKey = "lensControls.on" | "lensControls.off" | "lensControls.loading" | "lensControls.unknown";

function statusKey(enabled: boolean, loading: boolean, unknown: boolean): LensStatusKey {
  if (unknown) return "lensControls.unknown";
  if (loading) return "lensControls.loading";
  return enabled ? "lensControls.on" : "lensControls.off";
}

function useLensControlState({
  kind,
  policy,
  teamPolicy = null,
  teamPolicyStatus = "ready",
  canEdit,
  onPolicyChange,
}: LensControlsProps) {
  const { t } = useTranslation();
  const { data: service, isPending, isError } = $api.useQuery("get", "/lens/service", undefined, { staleTime: 30_000 });
  const [saving, setSaving] = useState(false);
  const enabled = effectiveLensEnabled(policy, teamPolicy, service?.configured ?? false);
  const inherited = kind === "key" && policy === null;
  const canReset = kind === "key" && policy !== null && canEdit;
  const inheritedUnavailable = inherited && teamPolicyStatus !== "ready";
  const connectionUnavailable = isPending || isError;
  const unavailable = saving || connectionUnavailable || inheritedUnavailable;
  const disabled = !canEdit || unavailable;
  const loadingPolicy = isPending || (inherited && teamPolicyStatus === "loading");
  const unknownTeamPolicy = inherited && teamPolicyStatus === "error";
  const unknownGlobalPolicy = policy === null && teamPolicy === null && isError;
  const status = statusKey(enabled, loadingPolicy, unknownTeamPolicy || unknownGlobalPolicy);
  const updatePolicy = async (next: LensPolicy): Promise<void> => {
    if (!canEdit || saving) return;
    setSaving(true);
    try {
      await onPolicyChange(next);
    } catch {
      toast.fromError(t("lensControls.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  return { service, isError, enabled, inherited, canReset, disabled, loadingPolicy, status, saving, updatePolicy };
}

export function LensControls(props: LensControlsProps) {
  const controlPlane = useControlPlane();
  const { t } = useTranslation();
  if (controlPlane) return <p>{t("controlPlane.actionUnavailable")}</p>;
  return <DataplaneLensControls {...props} />;
}

function DataplaneLensControls(props: LensControlsProps) {
  const { kind, href, teamPolicyStatus = "ready" } = props;
  const { t } = useTranslation();
  const { service, isError, enabled, inherited, canReset, disabled, loadingPolicy, status, saving, updatePolicy } =
    useLensControlState(props);
  const openLensContent = (
    <>
      {t("lensControls.open")}
      <ExternalLink className="size-4" />
    </>
  );

  return (
    <Card className="p-6">
      <div className="flex flex-wrap items-center gap-3">
        <LensSymbol size={28} />
        <h2 className="text-lg font-semibold">{t("lensControls.title")}</h2>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-sm text-muted-foreground">{t(status)}</span>
          <Switch
            aria-label={t("lensControls.title")}
            aria-busy={loadingPolicy || saving}
            checked={enabled}
            onCheckedChange={(checked) => void updatePolicy(checked)}
            disabled={disabled}
          />
          {href ? (
            <Button nativeButton={false} role="link" render={<Link href={href} />} className="ml-3">
              {openLensContent}
            </Button>
          ) : (
            <Button disabled className="ml-3">
              {openLensContent}
            </Button>
          )}
        </div>
      </div>
      <div className="ml-10 mt-3 flex items-center gap-3 text-xs text-muted-foreground">
        <span>{t(helperKey(kind, inherited))}</span>
        {canReset && (
          <Button
            variant="link"
            size="sm"
            className="h-auto p-0 text-xs"
            onClick={() => void updatePolicy(null)}
            disabled={saving || teamPolicyStatus !== "ready"}
          >
            {t("lensControls.useTeam")}
          </Button>
        )}
      </div>
      {inherited && teamPolicyStatus === "error" && (
        <p className="ml-10 mt-2 text-xs text-muted-foreground">{t("lensControls.teamUnavailable")}</p>
      )}
      {isError && <p className="ml-10 mt-2 text-xs text-muted-foreground">{t("lensControls.statusUnavailable")}</p>}
      {service && !service.configured && (
        <p className="ml-10 mt-2 text-xs text-muted-foreground">{t("lensControls.notConfigured")}</p>
      )}
    </Card>
  );
}

type LensCreateFieldProps = {
  policy: LensPolicy;
  teamPolicy?: LensPolicy;
  onChange: (enabled: boolean) => void;
};

export function LensCreateField(props: LensCreateFieldProps) {
  const controlPlane = useControlPlane();
  return controlPlane ? null : <DataplaneLensCreateField {...props} />;
}

function DataplaneLensCreateField({ policy, teamPolicy = null, onChange }: LensCreateFieldProps) {
  const { t } = useTranslation();
  const { data: service, isPending, isError } = $api.useQuery("get", "/lens/service", undefined, { staleTime: 30_000 });
  const enabled = effectiveLensEnabled(policy, teamPolicy, service?.configured ?? false);
  const status = statusKey(enabled, isPending, isError);
  return (
    <div className="mb-6 flex items-center gap-3 rounded-md border border-border p-3">
      <LensSymbol />
      <div>
        <p className="text-sm font-medium">{t("lensControls.title")}</p>
        <p className="text-xs text-muted-foreground">{t("lensControls.tracing")}</p>
        {isError && <p className="text-xs text-muted-foreground">{t("lensControls.statusUnavailable")}</p>}
      </div>
      <span className="ml-auto text-xs text-muted-foreground">{t(status)}</span>
      <Switch
        aria-label={t("lensControls.title")}
        checked={enabled}
        onCheckedChange={onChange}
        disabled={isPending || isError}
      />
    </div>
  );
}
