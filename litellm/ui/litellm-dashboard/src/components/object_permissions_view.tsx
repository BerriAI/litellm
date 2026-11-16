import React from "react";
import { Trans, useTranslation } from "react-i18next";
import VectorStorePermissions from "./permissions/VectorStorePermissions";
import MCPServerPermissions from "./permissions/MCPServerPermissions";
import AgentPermissions from "./permissions/AgentPermissions";
import type { ObjectPermission } from "./object_permission_types";
import { inheritedGrantTooltip, type InheritedGrant } from "./permissions/inheritedGrants";

interface ObjectPermissionsViewProps {
  objectPermission?: ObjectPermission | null;
  inheritedMcpServers?: InheritedGrant[];
  inheritedAgents?: InheritedGrant[];
  inheritedSkills?: InheritedGrant[];
  inheritedVectorStores?: InheritedGrant[];
  unresolvedAccessGroupIds?: string[] | null;
  variant?: "card" | "inline";
  className?: string;
  accessToken?: string | null;
}

export function ObjectPermissionsView({
  objectPermission,
  inheritedMcpServers = [],
  inheritedAgents = [],
  inheritedSkills,
  inheritedVectorStores,
  unresolvedAccessGroupIds,
  variant = "card",
  className = "",
  accessToken,
}: ObjectPermissionsViewProps) {
  const { t } = useTranslation();
  const vectorStores = objectPermission?.vector_stores || [];
  const mcpServers = objectPermission?.mcp_servers || [];
  const mcpAccessGroups = objectPermission?.mcp_access_groups || [];
  const mcpToolPermissions = objectPermission?.mcp_tool_permissions || {};
  const mcpToolsets = objectPermission?.mcp_toolsets || [];
  const agents = objectPermission?.agents || [];
  const agentAccessGroups = objectPermission?.agent_access_groups || [];
  const searchTools = objectPermission?.search_tools || [];
  const skills = objectPermission?.skills || [];
  const showsInheritedSkills = inheritedSkills !== undefined;
  const missingAccessGroupIds = unresolvedAccessGroupIds ?? [];
  const inheritedOnlySkills = (inheritedSkills ?? []).filter((grant) => !skills.includes(grant.id));

  const skillsContent = (() => {
    if (missingAccessGroupIds.length > 0) {
      return (
        <p className="mt-1 block text-xs break-words text-destructive">
          <Trans
            i18nKey="objectPermissionsView.skills.suspended"
            components={{ groups: <>{missingAccessGroupIds.join(", ")}</> }}
          />
        </p>
      );
    }
    if (skills.length === 0 && inheritedOnlySkills.length === 0) {
      return (
        <p className="mt-1 block text-xs text-muted-foreground">
          {showsInheritedSkills
            ? t("objectPermissionsView.skills.noneGranted")
            : t("objectPermissionsView.skills.noneGrantedDirectly")}
        </p>
      );
    }
    return (
      <p className="mt-1 block text-xs break-words text-foreground">
        {skills.join(", ")}
        {skills.length > 0 && inheritedOnlySkills.length > 0 ? ", " : ""}
        {inheritedOnlySkills.map((grant, index) => (
          <span key={grant.id} title={inheritedGrantTooltip(grant, t)} className="text-muted-foreground">
            {index > 0 ? ", " : ""}
            {grant.id}
            {t("objectPermissionsView.skills.accessGroupSuffix")}
          </span>
        ))}
      </p>
    );
  })();

  const content = (
    <div className={variant === "card" ? "grid grid-cols-1 @xl:grid-cols-2 @4xl:grid-cols-3 gap-6" : "space-y-4"}>
      <VectorStorePermissions
        vectorStores={vectorStores}
        inheritedVectorStores={inheritedVectorStores}
        accessToken={accessToken}
      />
      <MCPServerPermissions
        mcpServers={mcpServers}
        mcpAccessGroups={mcpAccessGroups}
        mcpToolPermissions={mcpToolPermissions}
        mcpToolsets={mcpToolsets}
        inheritedMcpServers={inheritedMcpServers}
        accessToken={accessToken}
      />
      <AgentPermissions
        agents={agents}
        agentAccessGroups={agentAccessGroups}
        inheritedAgents={inheritedAgents}
        accessToken={accessToken}
      />
      <div className="min-w-0 rounded-md border border-border p-4">
        <p className="text-sm font-medium text-foreground">{t("objectPermissionsView.searchTools.title")}</p>
        {searchTools.length === 0 ? (
          <p className="mt-1 block text-xs text-muted-foreground">
            {t("objectPermissionsView.searchTools.noRestriction")}
          </p>
        ) : (
          <p className="mt-1 block text-xs break-words text-foreground">{searchTools.join(", ")}</p>
        )}
      </div>
      <div className="min-w-0 rounded-md border border-border p-4">
        <p className="text-sm font-medium text-foreground">{t("objectPermissionsView.skills.title")}</p>
        {skillsContent}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`@container bg-card border border-border rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <div>
            <p className="font-semibold text-foreground">{t("objectPermissionsView.title")}</p>
            <p className="text-xs text-muted-foreground">{t("objectPermissionsView.subtitle")}</p>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <p className="font-medium text-foreground mb-3">{t("objectPermissionsView.title")}</p>
      {content}
    </div>
  );
}

export default ObjectPermissionsView;
