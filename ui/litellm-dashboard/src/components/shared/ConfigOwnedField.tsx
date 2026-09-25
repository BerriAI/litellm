import React from "react";

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import type { components } from "@/lib/http/schema";

export type FieldSource = components["schemas"]["RouterSettingsResponse"]["source"][string];
export type FieldSourceMap = Partial<Record<string, FieldSource>>;

export interface SourcesState {
  sessionKey: string;
  sources: FieldSourceMap | null;
  failed: boolean;
}

export const CONFIG_OWNED_MESSAGE = "Set in config.yaml and cannot be changed here";

export const isConfigOwned = (sources: FieldSourceMap | null | undefined, fieldName: string): boolean =>
  sources?.[fieldName] === "config";

interface ConfigOwnedFieldProps {
  frozen: boolean;
  children: React.ReactNode;
  className?: string;
}

export function ConfigOwnedField({ frozen, children, className = "inline-flex w-full" }: ConfigOwnedFieldProps) {
  if (!frozen) {
    return <span className={className}>{children}</span>;
  }
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger render={<span className={className} data-config-owned="true" />}>{children}</TooltipTrigger>
        <TooltipContent>{CONFIG_OWNED_MESSAGE}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
