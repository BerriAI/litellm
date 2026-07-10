import { useDisableUsageIndicator } from "@/app/(dashboard)/hooks/useDisableUsageIndicator";
import { useLicenseInfo } from "@/app/(dashboard)/hooks/license/useLicenseInfo";
import { getDaysUntilExpiration } from "@/utils/licenseUtils";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Meter, MeterIndicator, MeterLabel, MeterTrack } from "@/components/ui/meter";
import { useQuery } from "@tanstack/react-query";
import { Award, ChevronDown, Loader2 } from "lucide-react";
import { getRemainingUsers } from "./networking";

interface SidebarUsageCardProps {
  accessToken: string | null;
  collapsed: boolean;
  onExpandRail: () => void;
}

interface MeterData {
  label: string;
  used: number;
  total: number;
}

const formatExpiration = (daysRemaining: number | null): string => {
  if (daysRemaining === null) return "No expiration";
  if (daysRemaining < 0) return "Expired";
  if (daysRemaining === 0) return "Expires today";
  if (daysRemaining === 1) return "1 day remaining";
  if (daysRemaining < 30) return `${daysRemaining} days remaining`;
  if (daysRemaining < 60) return "1 month remaining";
  return `${Math.floor(daysRemaining / 30)} months remaining`;
};

const meterTone = (pct: number): "default" | "warning" | "over" => {
  if (pct > 100) return "over";
  if (pct >= 80) return "warning";
  return "default";
};

const UsageMeter = ({ label, used, total }: MeterData) => {
  const pct = total > 0 ? (used / total) * 100 : 0;
  return (
    <Meter value={used} max={total} aria-valuetext={`${used.toLocaleString()} of ${total.toLocaleString()}`}>
      <div className="flex items-baseline justify-between gap-2">
        <MeterLabel>{label}</MeterLabel>
        <span className="text-xs font-medium tabular-nums">
          <span className="text-foreground">{used.toLocaleString()}</span>
          <span className="text-muted-foreground"> / {total.toLocaleString()}</span>
        </span>
      </div>
      <MeterTrack>
        <MeterIndicator tone={meterTone(pct)} />
      </MeterTrack>
    </Meter>
  );
};

type RemainingUsage = NonNullable<Awaited<ReturnType<typeof getRemainingUsers>>>;

const remainingUsersQuery = (accessToken: string | null) => ({
  queryKey: ["sidebarRemainingUsers", accessToken] as const,
  queryFn: () => getRemainingUsers(accessToken as string),
  enabled: Boolean(accessToken),
  retry: false as const,
  staleTime: 5 * 60 * 1000,
});

const buildMeters = (data: RemainingUsage | null): MeterData[] => {
  if (!data) return [];
  return [
    ...(data.total_users != null ? [{ label: "Seats", used: data.total_users_used, total: data.total_users }] : []),
    ...(data.total_teams != null ? [{ label: "Teams", used: data.total_teams_used, total: data.total_teams }] : []),
  ];
};

/**
 * Bottom-dock "Enterprise usage" card for the sidebar. Backed only by data
 * LiteLLM actually exposes: seat (user) and team allocations from the license,
 * plus the license expiry. There is no plan-level spend or request cap, so the
 * design's Spend / API-request meters are intentionally omitted.
 */
export default function SidebarUsageCard({ accessToken, collapsed, onExpandRail }: SidebarUsageCardProps) {
  const disableUsageIndicator = useDisableUsageIndicator();
  const licenseInfo = useLicenseInfo(accessToken).data ?? null;
  const { data: usageData, isLoading } = useQuery(remainingUsersQuery(accessToken));
  const data = usageData ?? null;

  const hasData = data !== null && (data.total_users !== null || data.total_teams !== null);
  const noUsableData = !isLoading && !hasData;
  const noLicensedUsage = !licenseInfo?.has_license || noUsableData;
  if (disableUsageIndicator || !accessToken || noLicensedUsage) {
    return null;
  }

  if (collapsed) {
    return (
      <Button
        variant="outline"
        onClick={onExpandRail}
        title="Enterprise usage"
        className="h-9 w-full rounded-lg border-sidebar-border bg-sidebar text-sidebar-primary shadow-none hover:bg-sidebar-accent hover:text-sidebar-primary"
      >
        <Award className="size-[18px]" strokeWidth={1.75} />
      </Button>
    );
  }

  const daysUntilExpiration = licenseInfo?.expiration_date ? getDaysUntilExpiration(licenseInfo.expiration_date) : null;
  const subtitle = licenseInfo?.expiration_date ? formatExpiration(daysUntilExpiration) : "Active plan";
  const meters = buildMeters(data);

  return (
    <Collapsible defaultOpen className="overflow-hidden rounded-xl border border-sidebar-border bg-sidebar">
      <CollapsibleTrigger className="group/usage flex w-full items-center gap-2.5 px-3 py-2.5 text-left transition-colors hover:bg-sidebar-accent">
        <span className="flex size-[26px] flex-none items-center justify-center rounded-md bg-sidebar-primary/10 text-sidebar-primary">
          <Award className="size-4" strokeWidth={1.75} />
        </span>
        <span className="min-w-0 flex-1 leading-tight">
          <span className="block text-[13px] font-semibold text-foreground">Enterprise usage</span>
          <span className="block truncate text-[11px] text-muted-foreground">{subtitle}</span>
        </span>
        <ChevronDown className="size-4 flex-none -rotate-90 text-muted-foreground transition-transform group-data-[panel-open]/usage:rotate-0" />
      </CollapsibleTrigger>

      <CollapsibleContent className="flex flex-col gap-3 px-3 pt-0.5 pb-3">
        {isLoading && meters.length === 0 ? (
          <div className="flex items-center gap-2 py-1 text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" /> Loading…
          </div>
        ) : (
          meters.map((m) => <UsageMeter key={m.label} {...m} />)
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
