"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import {
  type DeploymentWarning,
  useDeploymentHealthAccess,
  useDeploymentWarnings,
  useUpdateDeploymentWarningIgnore,
} from "@/app/(dashboard)/hooks/deploymentWarnings/useDeploymentWarnings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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

const LOCAL_MINUTE_FORMAT = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const FULL_UTC_FORMAT = new Intl.DateTimeFormat("en-US", { dateStyle: "full", timeStyle: "long", timeZone: "UTC" });

const formatLocalMinute = (iso: string): string => LOCAL_MINUTE_FORMAT.format(new Date(iso));
const formatFullUtc = (iso: string): string => FULL_UTC_FORMAT.format(new Date(iso));

interface ItemPermissions {
  canManage: boolean;
  ignoreAvailable: boolean;
}

const IgnoreMenu = ({ warningId, title }: { warningId: string; title: string }) => {
  const update = useUpdateDeploymentWarningIgnore();
  const ignore = (snoozeSeconds?: number) =>
    update.mutate({ warningId, update: { ignored: true, snooze_seconds: snoozeSeconds } });
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={<Button variant="outline" size="sm" disabled={update.isPending} aria-label={`Ignore ${title}`} />}
      >
        Ignore
        <ChevronDown data-icon="inline-end" aria-hidden />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {SNOOZE_OPTIONS.map(({ label, seconds }) => (
          <DropdownMenuItem key={seconds} onClick={() => ignore(seconds)}>
            {label}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => ignore()}>Ignore forever</DropdownMenuItem>
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
  if (!canManage) return null;
  if (!warning.ignore) {
    return ignoreAvailable ? (
      <IgnoreMenu warningId={warning.id} title={title} />
    ) : (
      <p className="shrink-0 text-xs text-muted-foreground">Connect a database to ignore warnings</p>
    );
  }
  return warning.ignore.source === "admin" && ignoreAvailable ? (
    <RestoreButton warningId={warning.id} title={title} />
  ) : null;
};

const IgnoreStatus = ({ ignore }: { ignore: NonNullable<DeploymentWarning["ignore"]> }) => {
  if (ignore.source === "env_var") {
    return (
      <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline">Ignored</Badge>
        <span>
          by the <code className="font-mono">{ignore.env_var}</code> environment variable
        </span>
      </p>
    );
  }
  return (
    <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
      {ignore.ignored_until ? (
        <SimpleTooltip content={formatFullUtc(ignore.ignored_until)}>
          <Badge variant="outline">Snoozed until {formatLocalMinute(ignore.ignored_until)}</Badge>
        </SimpleTooltip>
      ) : (
        <Badge variant="outline">Ignored</Badge>
      )}
      <span>
        by {ignore.ignored_by} on {formatLocalMinute(ignore.ignored_at)}
      </span>
    </p>
  );
};

const WarningItem = ({ warning, ...permissions }: ItemPermissions & { warning: DeploymentWarning }) => {
  const content = deploymentWarningContent(warning.id);
  return (
    <article aria-label={content.title} className="flex flex-col gap-2 rounded-lg border border-border p-3">
      <div className="flex items-start justify-between gap-3">
        <p className="font-medium">{content.title}</p>
        <ItemAction warning={warning} title={content.title} {...permissions} />
      </div>
      {content.fix ? <p className="text-muted-foreground [&_code]:font-mono">{content.fix}</p> : null}
      {content.docsUrl ? (
        <a
          className="w-fit text-primary underline-offset-4 hover:underline"
          href={content.docsUrl}
          target="_blank"
          rel="noopener noreferrer"
        >
          Read the docs
        </a>
      ) : null}
      {warning.ignore ? <IgnoreStatus ignore={warning.ignore} /> : null}
    </article>
  );
};

const PanelBody = () => {
  const { canManage } = useDeploymentHealthAccess();
  const { data, isError } = useDeploymentWarnings({ refetchOnOpen: true });
  if (!data) {
    return (
      <p className="px-4 text-muted-foreground">
        {isError ? "Could not load deployment warnings" : "Loading deployment warnings"}
      </p>
    );
  }

  const permissions = { canManage, ignoreAvailable: data.ignore_available };
  const active = activeWarnings(data.warnings);
  const ignored = data.warnings.filter((warning) => warning.ignore);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto px-4 pb-4">
      {active.length === 0 ? <p className="text-muted-foreground">No active deployment warnings</p> : null}
      {SEVERITY_GROUPS.map(({ severity, label }) => {
        const group = active.filter((warning) => warning.severity === severity);
        return group.length === 0 ? null : (
          <section key={severity} aria-label={label} className="flex flex-col gap-3">
            <h3 className="text-sm font-medium">{label}</h3>
            {group.map((warning) => (
              <WarningItem key={warning.id} warning={warning} {...permissions} />
            ))}
          </section>
        );
      })}
      {ignored.length > 0 ? (
        <Collapsible className="flex flex-col gap-3">
          <CollapsibleTrigger
            render={<Button variant="ghost" size="sm" className="w-fit data-[panel-open]:[&>svg]:rotate-90" />}
          >
            <ChevronRight data-icon="inline-start" aria-hidden />
            Ignored and snoozed ({ignored.length})
          </CollapsibleTrigger>
          <CollapsibleContent className="flex flex-col gap-3">
            {ignored.map((warning) => (
              <WarningItem key={warning.id} warning={warning} {...permissions} />
            ))}
          </CollapsibleContent>
        </Collapsible>
      ) : null}
    </div>
  );
};

export const DeploymentHealthPanel = ({ handle }: { handle: DeploymentHealthHandle }) => (
  <Sheet handle={handle}>
    <SheetContent side="right" className="gap-0 data-[side=right]:sm:max-w-md">
      <SheetHeader>
        <SheetTitle>Deployment health</SheetTitle>
        <SheetDescription>Warnings about how this proxy is deployed</SheetDescription>
      </SheetHeader>
      <PanelBody />
    </SheetContent>
  </Sheet>
);
