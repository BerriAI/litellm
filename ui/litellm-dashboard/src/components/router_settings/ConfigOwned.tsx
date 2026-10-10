import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Lock } from "lucide-react";
import React from "react";

export type RouterSettingSource = "config" | "db" | "env" | "default" | "unset";
export type RouterSettingsSource = { [key: string]: RouterSettingSource };

export const CONFIG_OWNED_HINT = "Set in config.yaml. Edit the file to change it.";

export function isOwnedByConfig(source: RouterSettingsSource, key: string): boolean {
  return source[key] === "config";
}

export function ConfigOwnedBadge(): React.ReactElement {
  return (
    <Tooltip>
      <TooltipTrigger
        render={<Badge variant="outline" data-testid="config-owned-badge" className="ml-2 align-middle" />}
      >
        <Lock aria-hidden />
        config.yaml
      </TooltipTrigger>
      <TooltipContent>{CONFIG_OWNED_HINT}</TooltipContent>
    </Tooltip>
  );
}
