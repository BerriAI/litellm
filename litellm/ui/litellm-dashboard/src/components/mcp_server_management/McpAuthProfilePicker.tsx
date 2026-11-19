import React from "react";
import { useTranslation } from "react-i18next";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { MCPServer } from "@/components/mcp_tools/types";
import { grantedProfileFor, profileServers, reachableServers, withServerProfile } from "./mcpAuthProfileGrants";

interface McpAuthProfilePickerProps {
  readonly value: readonly string[] | null | undefined;
  readonly onChange: (grants: string[]) => void;
  readonly servers: readonly MCPServer[] | undefined;
  readonly followsTeam?: boolean;
}

export const useKeyAuthProfileServers = (
  teamId: string | null,
  userId: string | null,
  keyServerIds: readonly string[],
  keyToolsetIds: readonly string[],
): readonly MCPServer[] | undefined => {
  const { data: scoped } = useMCPServers(teamId, userId);
  const { data: all } = useMCPServers();
  const { data: toolsets } = useMCPToolsets();
  const { data: team } = useTeam(teamId ?? undefined);
  if (scoped === undefined || all === undefined) return undefined;
  const teamToolsetIds = team?.object_permission?.mcp_toolsets ?? [];
  return reachableServers({
    scoped,
    all,
    serverIds: keyServerIds,
    toolsets: toolsets ?? [],
    toolsetIds: [...keyToolsetIds, ...teamToolsetIds],
  });
};

const DEFAULT_PROFILE_VALUE = "default";

const McpAuthProfilePicker: React.FC<McpAuthProfilePickerProps> = ({
  value,
  onChange,
  servers: catalog,
  followsTeam = false,
}) => {
  const { t } = useTranslation();
  const idPrefix = React.useId();
  const servers = profileServers(catalog ?? []);
  const grants = value ?? [];
  if (servers.length === 0) return null;
  const noGrantLabel = followsTeam
    ? t("mcpServerManagement.authProfilePicker.teamOrDefaultProfile")
    : t("mcpServerManagement.authProfilePicker.defaultProfile");

  return (
    <div className="mb-6 space-y-3">
      <div>
        <p className="text-sm font-medium text-foreground">{t("mcpServerManagement.authProfilePicker.title")}</p>
        <p className="text-xs text-muted-foreground">{t("mcpServerManagement.authProfilePicker.description")}</p>
      </div>
      {servers.map((server) => {
        const granted = grantedProfileFor(grants, server);
        const items = [
          { value: DEFAULT_PROFILE_VALUE, label: noGrantLabel },
          ...(server.additional_auth ?? []).map((profile) => ({ value: profile.id, label: profile.id })),
        ];
        const selectId = [idPrefix, server.server_id].join("-");
        return (
          <div key={server.server_id} className="flex items-center justify-between gap-4">
            <Label htmlFor={selectId}>{server.alias || server.server_name || server.server_id}</Label>
            <Select
              value={granted ?? DEFAULT_PROFILE_VALUE}
              items={items}
              onValueChange={(next) =>
                onChange([...withServerProfile(grants, server, next === DEFAULT_PROFILE_VALUE ? null : next)])
              }
            >
              <SelectTrigger id={selectId} className="w-56">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {items.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        );
      })}
    </div>
  );
};

export default McpAuthProfilePicker;
