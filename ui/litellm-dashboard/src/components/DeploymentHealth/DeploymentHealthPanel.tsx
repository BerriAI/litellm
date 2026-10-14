"use client";

import { ChevronRight, CircleAlert, Ellipsis, ExternalLink, ShieldCheck, TriangleAlert } from "lucide-react";
import {
  type DeploymentWarning,
  useDeploymentHealthAccess,
  useDeploymentWarnings,
  useUpdateDeploymentWarningIgnore,
} from "@/app/(dashboard)/hooks/deploymentWarnings/useDeploymentWarnings";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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

const LOCAL_MINUTE_FORMAT = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const FULL_UTC_FORMAT = new Intl.DateTimeFormat("en-US", { dateStyle: "full", timeStyle: "long", timeZone: "UTC" });

const formatLocalMinute = (iso: string): string => LOCAL_MINUTE_FORMAT.format(new Date(iso));
const formatFullUtc = (iso: string): string => FULL_UTC_FORMAT.format(new Date(iso));

interface ItemPermissions {
  canManage: boolean;
  ignoreAvailable: boolean;
}

const ActionsMenu = ({ warning, title }: { warning: DeploymentWarning; title: string }) => {
  const update = useUpdateDeploymentWarningIgnore();
  const ignore = (snoozeSeconds?: number) =>
    update.mutate({ warningId: warning.id, update: { ignored: true, snooze_seconds: snoozeSeconds } });
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <Button
            variant="ghost"
            size="icon-sm"
            className="-my-1 size-7 text-muted-foreground"
            disabled={update.isPending}
            aria-label={`Actions for ${title}`}
          />
        }
      >
        <Ellipsis aria-hidden />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-44">
        {warning.ignore ? (
          <DropdownMenuItem onClick={() => update.mutate({ warningId: warning.id, update: { ignored: false } })}>
            Restore
          </DropdownMenuItem>
        ) : (
          <>
            {SNOOZE_OPTIONS.map(({ label, seconds }) => (
              <DropdownMenuItem key={seconds} onClick={() => ignore(seconds)}>
                {label}
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => ignore()}>Ignore forever</DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

const hasActions = (warning: DeploymentWarning, { canManage, ignoreAvailable }: ItemPermissions): boolean => {
  const adminControlled = !warning.ignore || warning.ignore.source === "admin";
  return canManage && ignoreAvailable && adminControlled;
};

const IgnoreStatus = ({ ignore }: { ignore: NonNullable<DeploymentWarning["ignore"]> }) => {
  if (ignore.source === "env_var") {
    return (
      <p className="text-[12px] leading-4 text-muted-foreground">
        <span>Ignored</span> by the <code className="font-mono">{ignore.env_var}</code> environment variable
      </p>
    );
  }
  return (
    <p className="text-[12px] leading-4 text-muted-foreground">
      {ignore.ignored_until ? (
        <SimpleTooltip content={formatFullUtc(ignore.ignored_until)}>
          <span className="underline decoration-dotted underline-offset-3">
            Snoozed until {formatLocalMinute(ignore.ignored_until)}
          </span>
        </SimpleTooltip>
      ) : (
        <span>Ignored</span>
      )}{" "}
      by {ignore.ignored_by} on {formatLocalMinute(ignore.ignored_at)}
    </p>
  );
};

const SeverityIcon = ({ warning }: { warning: DeploymentWarning }) => {
  const Icon = warning.severity === "critical" ? CircleAlert : TriangleAlert;
  const activeTone = warning.severity === "critical" ? "text-destructive" : "text-warning";
  const tone = warning.ignore ? "text-muted-foreground" : activeTone;
  return <Icon className={`mt-0.5 size-4 shrink-0 ${tone}`} aria-hidden />;
};

const WarningItem = ({ warning, ...permissions }: ItemPermissions & { warning: DeploymentWarning }) => {
  const content = deploymentWarningContent(warning.id);
  const showDatabaseHint = permissions.canManage && !permissions.ignoreAvailable && !warning.ignore;
  return (
    <article aria-label={content.title} className="flex gap-2 py-3">
      <SeverityIcon warning={warning} />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex items-start gap-2">
          <p className="min-w-0 flex-1 font-medium">{content.title}</p>
          {hasActions(warning, permissions) ? <ActionsMenu warning={warning} title={content.title} /> : null}
        </div>
        {content.fix && !warning.ignore ? (
          <p className="text-muted-foreground [&_code]:font-mono [&_code]:text-[12px] [&_code]:wrap-break-word [&_code]:text-foreground">
            {content.fix}
          </p>
        ) : null}
        {content.docsUrl && !warning.ignore ? (
          <a
            className="inline-flex w-fit items-center gap-1 font-medium text-foreground underline-offset-4 hover:underline"
            href={content.docsUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            Read the docs
            <ExternalLink className="size-3.5 text-muted-foreground" aria-hidden />
          </a>
        ) : null}
        {warning.ignore ? <IgnoreStatus ignore={warning.ignore} /> : null}
        {showDatabaseHint ? (
          <p className="text-[12px] leading-4 text-muted-foreground">Connect a database to ignore warnings</p>
        ) : null}
      </div>
    </article>
  );
};

const groupLabelClassName = "text-[12px] leading-4 font-medium text-muted-foreground";
const groupListClassName = "divide-y divide-border border-y border-border";

const PanelSkeleton = () => (
  <div className={groupListClassName} aria-hidden>
    {[0, 1].map((row) => (
      <div key={row} className="flex gap-2 py-3">
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
    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto px-4 pb-4 text-[13px] leading-5">
      {active.length === 0 ? (
        <div className="flex flex-col items-center gap-2 py-8 text-center">
          <ShieldCheck className="size-5 text-muted-foreground" aria-hidden />
          <p className="text-muted-foreground">No active deployment warnings</p>
        </div>
      ) : null}
      {SEVERITY_GROUPS.map(({ severity, label }) => {
        const group = active.filter((warning) => warning.severity === severity);
        return group.length === 0 ? null : (
          <section key={severity} aria-label={label} className="flex flex-col gap-2">
            <h3 className={groupLabelClassName}>{label}</h3>
            <div className={groupListClassName}>
              {group.map((warning) => (
                <WarningItem key={warning.id} warning={warning} {...permissions} />
              ))}
            </div>
          </section>
        );
      })}
      {ignored.length > 0 ? (
        <Collapsible className="flex flex-col gap-2">
          <CollapsibleTrigger
            className={`${groupLabelClassName} flex w-fit items-center gap-1 rounded-sm outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50 data-[panel-open]:[&>svg]:rotate-90`}
          >
            <ChevronRight className="size-3.5 transition-transform" aria-hidden />
            Ignored and snoozed ({ignored.length})
          </CollapsibleTrigger>
          <CollapsibleContent className={groupListClassName}>
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
      <SheetHeader className="gap-0.5 pr-12">
        <SheetTitle className="text-[15px] leading-[22px] font-semibold">Deployment health</SheetTitle>
        <SheetDescription className="text-[13px] leading-5">Warnings about how this proxy is deployed</SheetDescription>
      </SheetHeader>
      <PanelBody />
    </SheetContent>
  </Sheet>
);
