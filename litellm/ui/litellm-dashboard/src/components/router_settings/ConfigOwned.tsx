import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Lock } from "lucide-react";
import type { ParseKeys } from "i18next";
import React from "react";
import { useTranslation } from "react-i18next";

export type RouterSettingSource = "config" | "db" | "env" | "default" | "unset";
export type RouterSettingsSource = { [key: string]: RouterSettingSource };

export const CONFIG_OWNED_HINT: ParseKeys = "routerSettingsForm.configOwned.hint";
export const SOURCE_UNKNOWN_HINT: ParseKeys = "routerSettingsForm.configOwned.sourceUnknown";

export function isOwnedByConfig(source: RouterSettingsSource, key: string): boolean {
  return source[key] === "config";
}

export function ConfigOwnedBadge(): React.ReactElement {
  const { t } = useTranslation();
  return (
    <Tooltip>
      <TooltipTrigger
        render={<Badge variant="outline" data-testid="config-owned-badge" className="ml-2 align-middle" />}
      >
        <Lock aria-hidden />
        config.yaml
      </TooltipTrigger>
      <TooltipContent>{t(CONFIG_OWNED_HINT)}</TooltipContent>
    </Tooltip>
  );
}
