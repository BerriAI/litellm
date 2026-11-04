"use client";

import {
  BellOff,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Clock,
  ExternalLink,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import {
  type DeploymentWarning,
  useDeploymentHealthAccess,
  useDeploymentWarnings,
  useUpdateDeploymentWarningIgnore,
} from "@/app/(dashboard)/hooks/deploymentWarnings/useDeploymentWarnings";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { cn } from "@/lib/cva.config";
import { Skeleton } from "@/components/ui/skeleton";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { activeWarnings, type DeploymentHealthHandle } from "./DeploymentHealth";
import { deploymentWarningContent } from "./DeploymentWarningContent";

const DAY_SECONDS = 24 * 60 * 60;

const SNOOZE_OPTIONS = [
  { label: "Snooze for 1 day", seconds: DAY_SECONDS },
  { label: "Snooze for 7 days", seconds: 7 * DAY_SECONDS },
  { label: "Snooze for 30 days", seconds: 30 * DAY_SECONDS },
] as const;

const SEVERITY_GROUPS = [
  { severity: "critical", label: "Critical" },
  { severity: "warning", label: "Warnings" },
] as const;

const LOCAL_DAY_FORMAT = new Intl.DateTimeFormat(undefined, { dateStyle: "medium" });
const LOCAL_MINUTE_FORMAT = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const FULL_UTC_FORMAT = new Intl.DateTimeFormat("en-US", { dateStyle: "full", timeStyle: "long", timeZone: "UTC" });

const formatLocalMinute = (iso: string): string => LOCAL_MINUTE_FORMAT.format(new Date(iso));
const formatLocalDay = (iso: string): string => LOCAL_DAY_FORMAT.format(new Date(iso));
const formatFullUtc = (iso: string): string => FULL_UTC_FORMAT.format(new Date(iso));

interface ItemPermissions {
  canManage: boolean;
  ignoreAvailable: boolean;
}

type IgnoreState = NonNullable<DeploymentWarning["ignore"]>;

const SnoozeMenu = ({ warningId, title }: { warningId: string; title: string }) => {
  const update = useUpdateDeploymentWarningIgnore();
  const ignore = (snoozeSeconds?: number) =>
    update.mutate({ warningId, update: { ignored: true, snooze_seconds: snoozeSeconds } });
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-[13px]"
            disabled={update.isPending}
            aria-label={`Snooze ${title}`}
          />
        }
      >
        Snooze
        <ChevronDown data-icon="inline-end" aria-hidden />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-48 [&_[data-slot=dropdown-menu-item]]:whitespace-nowrap">
        {SNOOZE_OPTIONS.map(({ label, seconds }) => (
          <DropdownMenuItem key={seconds} onClick={() => ignore(seconds)}>
            <Clock aria-hidden />
            {label}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => ignore()}>
          <BellOff aria-hidden />
          Ignore forever
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

const RestoreButton = ({ warningId, title }: { warningId: string; title: string }) => {
  const update = useUpdateDeploymentWarningIgnore();
  return (
    <Button
      variant="outline"
      size="sm"
      className="h-7 text-[13px]"
      disabled={update.isPending}
      aria-label={`Restore ${title}`}
      onClick={() => update.mutate({ warningId, update: { ignored: false } })}
    >
      Restore
    </Button>
  );
};

const ItemAction = ({
  warning,
  title,
  canManage,
  ignoreAvailable,
}: ItemPermissions & { warning: DeploymentWarning; title: string }) => {
  if (!canManage || !ignoreAvailable) return null;
  if (!warning.ignore) return <SnoozeMenu warningId={warning.id} title={title} />;
  return warning.ignore.source === "admin" ? <RestoreButton warningId={warning.id} title={title} /> : null;
};

const IgnoreMeta = ({ ignore }: { ignore: IgnoreState }) => {
  if (ignore.source === "env_var") {
    return (
      <span>
        By env var <code className="font-mono wrap-anywhere">{ignore.env_var}</code>
      </span>
    );
  }
  if (ignore.ignored_until) {
    return (
      <span>
        <SimpleTooltip content={formatFullUtc(ignore.ignored_until)}>
          <span className="underline decoration-dotted underline-offset-3">
            Until {formatLocalMinute(ignore.ignored_until)}
          </span>
        </SimpleTooltip>{" "}
        · by {ignore.ignored_by}
      </span>
    );
  }
  return (
    <span>
      Since {formatLocalDay(ignore.ignored_at)} · by {ignore.ignored_by}
    </span>
  );
};

const IgnoreStatus = ({ ignore }: { ignore: IgnoreState }) => {
  const snoozed = ignore.source === "admin" && Boolean(ignore.ignored_until);
  const Icon = snoozed ? Clock : BellOff;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[12px] leading-4 text-muted-foreground">
      <Badge variant="outline" className="gap-1 bg-muted text-muted-foreground">
        <Icon aria-hidden />
        {snoozed ? "Snoozed" : "Ignored"}
      </Badge>
      <IgnoreMeta ignore={ignore} />
    </div>
  );
};

const WarningItem = ({ warning, ...permissions }: ItemPermissions & { warning: DeploymentWarning }) => {
  const content = deploymentWarningContent(warning.id);
  const critical = warning.severity === "critical";
  const SeverityIcon = critical ? CircleAlert : TriangleAlert;
  return (
    <Alert
      variant={critical ? "error" : "warning"}
      role="article"
      aria-label={content.title}
      data-ignored={warning.ignore ? "" : undefined}
      className="gap-x-2 border-warning/25 *:[svg]:text-warning data-[variant=error]:*:[svg]:text-destructive bg-warning/6 p-3 text-[13px] leading-5 text-foreground *:data-[slot=alert-description]:text-muted-foreground data-[variant=error]:border-destructive/35 data-[variant=error]:bg-destructive/10 data-ignored:border-warning/20 data-ignored:bg-warning/4 data-ignored:data-[variant=error]:border-destructive/20 data-ignored:data-[variant=error]:bg-destructive/4"
    >
      <SeverityIcon aria-hidden />
      <AlertTitle>{content.title}</AlertTitle>
      <AlertDescription className="flex flex-col [&_a]:no-underline [&_a]:hover:underline [&_p:not(:last-child)]:mb-0 text-[13px] leading-5 text-pretty [&_code]:font-mono [&_code]:text-[12px] [&_code]:wrap-anywhere [&_code]:text-foreground">
        {content.fix ? <p className="mt-0.5">{content.fix}</p> : null}
        {warning.ignore ? <IgnoreStatus ignore={warning.ignore} /> : null}
        <div className="mt-2 flex items-center gap-2">
          {content.docsUrl ? (
            <a
              href={content.docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                buttonVariants({ variant: "link" }),
                "h-auto gap-1 p-0 text-[13px] leading-5 font-medium text-foreground no-underline hover:underline",
              )}
            >
              Read the docs
              <ExternalLink className="size-3.5 text-muted-foreground" aria-hidden />
            </a>
          ) : null}
          <span className="flex-1" />
          <ItemAction warning={warning} title={content.title} {...permissions} />
        </div>
      </AlertDescription>
    </Alert>
  );
};

const GroupTrigger = ({ label, count }: { label: string; count: number }) => (
  <CollapsibleTrigger className="flex h-8 w-fit items-center gap-1.5 rounded-sm font-semibold outline-none focus-visible:ring-3 focus-visible:ring-ring/50 data-[panel-open]:[&>svg]:rotate-90">
    <ChevronRight className="size-3.5 text-muted-foreground transition-transform" aria-hidden />
    {label} <span className="font-normal text-muted-foreground">{count}</span>
  </CollapsibleTrigger>
);

const WarningGroup = ({
  label,
  warnings,
  defaultOpen,
  permissions,
}: {
  label: string;
  warnings: readonly DeploymentWarning[];
  defaultOpen: boolean;
  permissions: ItemPermissions;
}) => (
  <Collapsible defaultOpen={defaultOpen} render={<section aria-label={label} />} className="flex flex-col gap-2">
    <GroupTrigger label={label} count={warnings.length} />
    <CollapsibleContent className="flex flex-col gap-2">
      {warnings.map((warning) => (
        <WarningItem key={warning.id} warning={warning} {...permissions} />
      ))}
    </CollapsibleContent>
  </Collapsible>
);

const PanelSkeleton = () => (
  <div className="flex flex-col gap-2" aria-hidden>
    {[0, 1].map((row) => (
      <div key={row} className="flex gap-2 rounded-lg border border-border p-3">
        <Skeleton className="size-4 rounded-full" />
        <div className="flex flex-1 flex-col gap-2">
          <Skeleton className="h-3 w-2/3" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-5/6" />
        </div>
      </div>
    ))}
  </div>
);

const NoActiveWarnings = ({ hasIgnored }: { hasIgnored: boolean }) => (
  <Empty className="gap-2 p-0 py-6">
    <EmptyHeader>
      <EmptyMedia
        variant="icon"
        className="border border-border bg-transparent text-muted-foreground [&_svg:not([class*='size-'])]:size-4"
      >
        <ShieldCheck aria-hidden />
      </EmptyMedia>
      <EmptyTitle className="text-[15px] leading-[22px] font-semibold tracking-normal">No active warnings</EmptyTitle>
      <EmptyDescription className="text-[13px] leading-5">
        {hasIgnored
          ? "Snoozed and ignored warnings are listed below"
          : "Deployment warnings show up here when this proxy\u2019s setup needs attention"}
      </EmptyDescription>
    </EmptyHeader>
  </Empty>
);

const PanelBody = () => {
  const { canManage } = useDeploymentHealthAccess();
  const { data, isError } = useDeploymentWarnings({ refetchOnOpen: true });
  if (!data) {
    return (
      <div className="flex flex-col gap-2 px-4 pb-4 text-[13px] leading-5">
        {isError ? (
          <p className="text-destructive">Could not load deployment warnings</p>
        ) : (
          <>
            <span className="sr-only">Loading deployment warnings</span>
            <PanelSkeleton />
          </>
        )}
      </div>
    );
  }

  const permissions = { canManage, ignoreAvailable: data.ignore_available };
  const active = activeWarnings(data.warnings);
  const ignored = data.warnings.filter((warning) => warning.ignore);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pb-4 text-[13px] leading-5">
      {active.length === 0 ? <NoActiveWarnings hasIgnored={ignored.length > 0} /> : null}
      {SEVERITY_GROUPS.map(({ severity, label }) => {
        const group = active.filter((warning) => warning.severity === severity);
        return group.length === 0 ? null : (
          <WarningGroup key={severity} label={label} warnings={group} defaultOpen permissions={permissions} />
        );
      })}
      {ignored.length > 0 ? (
        <WarningGroup label="Ignored and snoozed" warnings={ignored} defaultOpen={false} permissions={permissions} />
      ) : null}
    </div>
  );
};

export const DeploymentHealthPanel = ({ handle }: { handle: DeploymentHealthHandle }) => (
  <Sheet handle={handle}>
    <SheetContent side="right" className="gap-0 data-[side=right]:sm:max-w-md">
      <SheetHeader className="gap-0.5 pr-12">
        <SheetTitle className="text-[15px] leading-[22px] font-semibold">Deployment health</SheetTitle>
        <SheetDescription className="text-[13px] leading-5">Warnings about how this proxy is deployed</SheetDescription>
      </SheetHeader>
      <PanelBody />
    </SheetContent>
  </Sheet>
);
