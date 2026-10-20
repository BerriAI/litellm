"use client";

import { createContext, useContext, type ReactNode } from "react";
import { Waypoints } from "lucide-react";

import { useAutoRouterModelGroups } from "@/app/(dashboard)/hooks/models/useModels";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";

const NO_AUTO_ROUTERS: ReadonlySet<string> = new Set<string>();

const AutoRouterModelGroupsContext = createContext<ReadonlySet<string>>(NO_AUTO_ROUTERS);

export function AutoRouterModelGroupsProvider({ children }: { children: ReactNode }) {
  const autoRouterModelGroups = useAutoRouterModelGroups();

  return (
    <AutoRouterModelGroupsContext.Provider value={autoRouterModelGroups}>
      {children}
    </AutoRouterModelGroupsContext.Provider>
  );
}

export function useIsAutoRoutedModelGroup(modelGroup?: string | null): boolean {
  const autoRouterModelGroups = useContext(AutoRouterModelGroupsContext);

  return Boolean(modelGroup) && autoRouterModelGroups.has(modelGroup as string);
}

export function AutoRouterIcon({ size = 12, className }: { size?: number; className?: string }) {
  return <Waypoints size={size} className={className} aria-hidden />;
}

export interface AutoRouterTagProps {
  modelGroup?: string | null;
  className?: string;
}

export function AutoRouterTag({ modelGroup, className }: AutoRouterTagProps) {
  const isAutoRouted = useIsAutoRoutedModelGroup(modelGroup);

  if (!isAutoRouted) return null;

  return (
    <Badge
      variant="secondary"
      title={`Routed by auto-router "${modelGroup}"`}
      className={cn("gap-1.5 px-2.5 py-1 text-sm font-normal text-foreground", className)}
    >
      <Waypoints aria-hidden />
      {modelGroup}
    </Badge>
  );
}
