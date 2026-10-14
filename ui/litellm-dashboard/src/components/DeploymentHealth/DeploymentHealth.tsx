"use client";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { CircleAlert, Shield } from "lucide-react";
import {
  type DeploymentWarning,
  useDeploymentHealthAccess,
  useDeploymentWarnings,
} from "@/app/(dashboard)/hooks/deploymentWarnings/useDeploymentWarnings";
import { Button } from "@/components/ui/button";
import { SheetTrigger } from "@/components/ui/sheet";
import { deploymentWarningContent } from "./DeploymentWarningContent";

export type DeploymentHealthHandle = DialogPrimitive.Handle<unknown>;

export const createDeploymentHealthHandle = (): DeploymentHealthHandle => DialogPrimitive.createHandle();

export const activeWarnings = (warnings: readonly DeploymentWarning[]): DeploymentWarning[] =>
  warnings.filter((warning) => !warning.ignore);

const pluralWarnings = (count: number): string => `${count} active ${count === 1 ? "warning" : "warnings"}`;

export const DeploymentHealthIcon = ({ handle }: { handle: DeploymentHealthHandle }) => {
  const { canView } = useDeploymentHealthAccess();
  const { data } = useDeploymentWarnings();
  if (!canView) return null;

  const active = activeWarnings(data?.warnings ?? []);
  const hasCritical = active.some((warning) => warning.severity === "critical");

  return (
    <SheetTrigger
      handle={handle}
      render={<Button variant="ghost" size="icon" className="relative text-muted-foreground hover:text-foreground" />}
      aria-label={active.length > 0 ? `Deployment health, ${pluralWarnings(active.length)}` : "Deployment health"}
    >
      <Shield className="size-4" aria-hidden />
      {active.length > 0 ? (
        <span
          data-tone={hasCritical ? "error" : "neutral"}
          className="absolute top-1 left-5 flex h-4 min-w-4 items-center justify-center rounded-full px-1 text-[12px] leading-4 font-medium tabular-nums ring-2 ring-background data-[tone=error]:bg-destructive data-[tone=error]:text-white data-[tone=neutral]:bg-muted-foreground data-[tone=neutral]:text-background"
          aria-hidden
        >
          {active.length}
        </span>
      ) : null}
    </SheetTrigger>
  );
};

const bannerLinkClassName = "h-auto p-0 text-[13px] leading-5 font-medium text-foreground underline underline-offset-3";

export const DeploymentHealthCriticalRow = ({ handle }: { handle: DeploymentHealthHandle }) => {
  const { canView } = useDeploymentHealthAccess();
  const { data } = useDeploymentWarnings();
  const critical = activeWarnings(data?.warnings ?? []).filter((warning) => warning.severity === "critical");
  if (!canView || critical.length === 0) return null;

  const [top, ...rest] = critical;
  return (
    <div
      role="alert"
      aria-label="Deployment alert"
      className="flex min-h-10 shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b border-destructive/25 bg-destructive/10 px-4 py-2 text-[13px] leading-5 text-foreground"
    >
      <CircleAlert className="size-4 shrink-0 text-destructive" aria-hidden />
      <strong className="min-w-0 font-medium">{deploymentWarningContent(top.id).title}</strong>
      <SheetTrigger handle={handle} render={<Button variant="link" className={bannerLinkClassName} />}>
        Review
      </SheetTrigger>
      {rest.length > 0 ? (
        <SheetTrigger
          handle={handle}
          render={
            <Button
              variant="link"
              className="ml-auto h-auto p-0 text-[12px] leading-4 font-normal text-muted-foreground hover:text-foreground"
            />
          }
        >
          +{rest.length} more
        </SheetTrigger>
      ) : null}
    </div>
  );
};
