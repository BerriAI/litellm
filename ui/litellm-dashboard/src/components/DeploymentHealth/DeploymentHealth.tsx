"use client";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { CircleAlert, Shield, TriangleAlert } from "lucide-react";
import {
  type DeploymentWarning,
  useDeploymentHealthAccess,
  useDeploymentWarnings,
} from "@/app/(dashboard)/hooks/deploymentWarnings/useDeploymentWarnings";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { Badge } from "@/components/ui/badge";
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
        <Badge
          data-tone={hasCritical ? "error" : "neutral"}
          className="absolute top-1 left-5 h-4 border-0 min-w-4 px-1 tabular-nums ring-2 ring-background data-[tone=error]:bg-destructive data-[tone=error]:text-white data-[tone=neutral]:bg-muted-foreground data-[tone=neutral]:text-background"
          aria-hidden
        >
          {active.length}
        </Badge>
      ) : null}
    </SheetTrigger>
  );
};

export const DeploymentHealthBanner = ({ handle }: { handle: DeploymentHealthHandle }) => {
  const { canView } = useDeploymentHealthAccess();
  const { data } = useDeploymentWarnings();
  const active = activeWarnings(data?.warnings ?? []);
  const top = active.find((warning) => warning.severity === "critical") ?? active[0];
  if (!canView || !top) return null;

  const critical = top.severity === "critical";
  const Icon = critical ? CircleAlert : TriangleAlert;
  return (
    <Alert
      variant={critical ? "error" : "warning"}
      role={critical ? "alert" : "status"}
      aria-label="Deployment alert"
      className="min-h-10 shrink-0 items-center *:[svg]:text-warning data-[variant=error]:*:[svg]:text-destructive rounded-none border-x-0 border-t-0 border-b-warning/25 bg-warning/10 px-4 py-2 text-[13px] leading-5 text-foreground data-[variant=error]:border-b-destructive/25 data-[variant=error]:bg-destructive/10"
    >
      <Icon aria-hidden />
      <AlertTitle className="flex flex-wrap items-center gap-x-2">
        {deploymentWarningContent(top.id).title}
        <SheetTrigger
          handle={handle}
          render={
            <Button variant="link" className="h-auto p-0 text-[13px] leading-5 font-medium text-foreground underline" />
          }
        >
          Review
        </SheetTrigger>
      </AlertTitle>
    </Alert>
  );
};
