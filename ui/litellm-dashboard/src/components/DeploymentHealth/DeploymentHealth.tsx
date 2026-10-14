"use client";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { CircleAlert, Shield } from "lucide-react";
import {
  type DeploymentWarning,
  useDeploymentHealthAccess,
  useDeploymentWarnings,
} from "@/app/(dashboard)/hooks/deploymentWarnings/useDeploymentWarnings";
import { Alert, AlertAction, AlertTitle } from "@/components/shared/Alert";
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
      render={<Button variant="ghost" size={active.length > 0 ? "sm" : "icon"} />}
      aria-label={active.length > 0 ? `Deployment health, ${pluralWarnings(active.length)}` : "Deployment health"}
    >
      <Shield aria-hidden />
      {active.length > 0 ? (
        <Badge variant={hasCritical ? "destructive" : "secondary"} aria-hidden>
          {active.length}
        </Badge>
      ) : null}
    </SheetTrigger>
  );
};

export const DeploymentHealthCriticalRow = ({ handle }: { handle: DeploymentHealthHandle }) => {
  const { canView } = useDeploymentHealthAccess();
  const { data } = useDeploymentWarnings();
  const critical = activeWarnings(data?.warnings ?? []).filter((warning) => warning.severity === "critical");
  if (!canView || critical.length === 0) return null;

  const [top, ...rest] = critical;
  return (
    <Alert variant="error" className="rounded-none border-x-0 border-t-0">
      <CircleAlert aria-hidden />
      <AlertTitle>
        {deploymentWarningContent(top.id).title}
        {rest.length > 0 ? <span className="ml-2 font-normal">+{rest.length} more</span> : null}
      </AlertTitle>
      <AlertAction>
        <SheetTrigger handle={handle} render={<Button variant="outline" size="sm" />}>
          Review
        </SheetTrigger>
      </AlertAction>
    </Alert>
  );
};
