import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { mapEmptyStringToNull } from "@/utils/keyUpdateUtils";
import { ArrowLeft } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { EntityLink } from "@/components/shared/EntityLink";
import { teamDetailHref } from "@/utils/entityLinks";
import { KeyInfoHeader } from "./KeyInfoHeader";
import KeySavingsTab from "./KeySavingsTab";
import { useEffect, useState } from "react";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam, rolesWithWriteAccess } from "../../utils/roles";
import { mapDisplayToInternalNames, mapInternalToDisplayNames } from "../callback_info_helpers";
import AutoRotationView from "../common_components/AutoRotationView";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import RouterSettingsSummary from "../common_components/RouterSettingsSummary";
import { hasRouterSettings } from "../common_components/routerSettingsPayload";
import { extractLoggingSettings, formatMetadataForDisplay, stripTagsFromMetadata } from "../key_info_utils";
import { KeyResponse } from "../key_team_helpers/key_list";
import LoggingSettingsView from "../logging_settings_view";
import { toast } from "@/lib/toast";
import { getPolicyInfoWithGuardrails, keyDeleteCall, keyUpdateCall } from "../networking";
import { useResetKeySpend } from "@/app/(dashboard)/hooks/keys/useResetKeySpend";
import { useSetKeyBlockedState } from "@/app/(dashboard)/hooks/keys/useSetKeyBlockedState";
import { keyKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useQueryClient } from "@tanstack/react-query";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { extractMcpEntitlement } from "../mcp_server_management/mcpEntitlement";
import ObjectPermissionsView from "../object_permissions_view";
import { RegenerateKeyModal } from "../organisms/RegenerateKeyModal";
import { parseErrorMessage } from "../shared/errorUtils";
import { InheritedBudgetHint, inheritedBudgetGates } from "../shared/InheritedBudgetHint";
import { KeyEditView } from "./key_edit_view";

interface KeyInfoViewProps {
  keyId: string;
  onClose: () => void;
  keyData: KeyResponse | undefined;
  onKeyDataUpdate?: (data: Partial<KeyResponse>) => void;
  onDelete?: () => void;
  teams: any[] | null;
  backButtonText?: string;
}

// Premium fields (from LiteLLM_ManagementEndpoint_MetadataFields_Premium in
// litellm/proxy/_types.py) that the key-edit form submits as arrays/strings, where
// "empty" means "unset". The loop below drops them when they're empty-and-were-empty
// so a non-premium edit of unrelated fields doesn't trip the server's premium gate.
//
// Boolean premium fields (e.g. disable_global_guardrails) do NOT belong here: false is
// a real value, not "empty", so isEmptyValue(false) is false and the loop would never
// drop it — we'd resend false on every edit and trip the gate. Booleans get their own
// "send only when changed" guard instead (see disable_global_guardrails below).
const PREMIUM_METADATA_FIELDS = ["policies", "guardrails", "prompts", "tags", "allowed_passthrough_routes"] as const;

const isEmptyValue = (v: unknown): boolean =>
  v == null || (Array.isArray(v) && v.length === 0) || (typeof v === "string" && v.trim() === "");

/**
 * ─────────────────────────────────────────────────────────────────────────
 * @deprecated
 * This component is being DEPRECATED in favor of src/app/(dashboard)/virtual-keys/components/KeyInfoView.tsx
 * Please contribute to the new refactor.
 * ─────────────────────────────────────────────────────────────────────────
 */
export default function KeyInfoView({
  onClose,
  keyData,
  teams,
  onKeyDataUpdate,
  onDelete,
  backButtonText = "Back to Keys",
}: KeyInfoViewProps) {
  const { accessToken, userId: userID, userRole, premiumUser } = useAuthorized();
  const queryClient = useQueryClient();
  const canEditGuardrails = premiumUser || (userRole != null && rolesWithWriteAccess.includes(userRole));
  const { teams: teamsData } = useTeams();
  const { data: organizations } = useOrganizations();
  const { data: projects } = useProjects();
  const { data: uiSettingsData } = useUISettings();
  const { data: allMcpServers } = useMCPServers();
  const { data: allMcpToolsets } = useMCPToolsets();
  const enableProjectsUI = Boolean(uiSettingsData?.values?.enable_projects_ui);
  const [isEditing, setIsEditing] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [isRegenerateModalOpen, setIsRegenerateModalOpen] = useState(false);
  const [isResetSpendModalOpen, setIsResetSpendModalOpen] = useState(false);
  const [isBlockModalOpen, setIsBlockModalOpen] = useState(false);
  const { mutate: resetKeySpend, isPending: resetSpendLoading } = useResetKeySpend();
  const { mutate: setKeyBlockedState, isPending: blockLoading } = useSetKeyBlockedState();
  // Add local state to maintain key data and track regeneration
  const [currentKeyData, setCurrentKeyData] = useState<KeyResponse | undefined>(keyData);
  const [lastRegeneratedAt, setLastRegeneratedAt] = useState<Date | null>(null);
  const [keyDataUpdateHeldUntilModalClose, setKeyDataUpdateHeldUntilModalClose] = useState<Partial<KeyResponse> | null>(
    null,
  );
  const [isRecentlyRegenerated, setIsRecentlyRegenerated] = useState(false);
  const [policyGuardrails, setPolicyGuardrails] = useState<Record<string, string[]>>({});
  const [loadingPolicies, setLoadingPolicies] = useState(false);

  // Update local state when keyData prop changes (but don't reset to undefined)
  useEffect(() => {
    if (keyData) {
      setCurrentKeyData(keyData);
    }
  }, [keyData]);

  // Fetch resolved guardrails for all policies
  useEffect(() => {
    const fetchPolicyGuardrails = async () => {
      const policies = currentKeyData?.metadata?.policies;
      if (!accessToken || !policies || !Array.isArray(policies) || policies.length === 0) {
        return;
      }

      setLoadingPolicies(true);
      const guardrailsMap: Record<string, string[]> = {};

      try {
        await Promise.all(
          policies.map(async (policyName: string) => {
            try {
              const policyInfo = await getPolicyInfoWithGuardrails(accessToken, policyName);
              guardrailsMap[policyName] = policyInfo.resolved_guardrails || [];
            } catch (error) {
              console.error(`Failed to fetch guardrails for policy ${policyName}:`, error);
              guardrailsMap[policyName] = [];
            }
          }),
        );
        setPolicyGuardrails(guardrailsMap);
      } catch (error) {
        console.error("Failed to fetch policy guardrails:", error);
      } finally {
        setLoadingPolicies(false);
      }
    };

    fetchPolicyGuardrails();
  }, [accessToken, currentKeyData?.metadata?.policies]);

  // Reset recent regeneration indicator after 5 seconds
  useEffect(() => {
    if (isRecentlyRegenerated) {
      const timer = setTimeout(() => {
        setIsRecentlyRegenerated(false);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [isRecentlyRegenerated]);

  // Use currentKeyData instead of keyData throughout the component
  if (!currentKeyData) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeft className="size-4" />
          {backButtonText}
        </Button>
        <p className="text-sm">Key not found</p>
      </div>
    );
  }

  const handleKeyUpdate = async (formValues: Record<string, any>) => {
    try {
      if (!accessToken) return;

      const currentKey = formValues.token;
      formValues.key = currentKey;

      // Guard premium features
      if (!canEditGuardrails) {
        delete formValues.guardrails;
        delete formValues.prompts;
      }

      // Drop premium metadata fields that are empty AND were empty before.
      // The /key/update response echoes defaults like `policies: []` back into
      // state; without this, the next save resends `[]` and trips the premium
      // gate in prepare_metadata_fields for non-premium users.
      for (const field of PREMIUM_METADATA_FIELDS) {
        const previousValue =
          (currentKeyData.metadata as Record<string, unknown> | undefined)?.[field] ??
          (currentKeyData as unknown as Record<string, unknown>)[field];
        if (isEmptyValue(formValues[field]) && isEmptyValue(previousValue)) {
          delete formValues[field];
        }
      }

      // disable_global_guardrails is premium-gated server-side; only send it when it
      // changed so a non-premium edit of unrelated fields isn't blocked by that gate.
      const previousDisableGlobalGuardrails = Boolean(
        (currentKeyData.metadata as Record<string, unknown> | undefined)?.disable_global_guardrails,
      );
      if (Boolean(formValues.disable_global_guardrails) === previousDisableGlobalGuardrails) {
        delete formValues.disable_global_guardrails;
      }

      // Handle max budget empty string
      formValues.max_budget = mapEmptyStringToNull(formValues.max_budget);

      // Handle object_permission updates
      if (formValues.vector_stores !== undefined) {
        formValues.object_permission = {
          ...currentKeyData.object_permission,
          vector_stores: formValues.vector_stores || [],
        };
        // Remove vector_stores from the top level as it should be in object_permission
        delete formValues.vector_stores;
      }

      const mcpEntitlement = extractMcpEntitlement(formValues, allMcpServers ?? [], allMcpToolsets ?? []);
      if (mcpEntitlement) {
        // Without a catalog the grants an allowlist key still has are unresolvable, so nothing is
        // pruned and a revocation would save as a no-op while reporting success. Refuse instead.
        const unresolvableSelection =
          allMcpServers === undefined ||
          mcpEntitlement.mcp_toolsets.some(
            (toolsetId) => !(allMcpToolsets ?? []).some((toolset) => toolset.toolset_id === toolsetId),
          );
        if (unresolvableSelection && Object.keys(mcpEntitlement.mcp_tool_permissions).length > 0) {
          toast.error("MCP server or toolset list is unavailable, so MCP permissions cannot be saved yet. Retry.");
          return;
        }
        formValues.object_permission = {
          ...(formValues.object_permission ?? currentKeyData.object_permission),
          ...mcpEntitlement,
        };
      }
      delete formValues.mcp_servers_and_groups;
      delete formValues.mcp_tool_permissions;

      // Handle agent permissions
      if (formValues.agents_and_groups !== undefined) {
        const { agents, accessGroups } = formValues.agents_and_groups || { agents: [], accessGroups: [] };
        formValues.object_permission = {
          ...formValues.object_permission,
          agents: agents || [],
          agent_access_groups: accessGroups || [],
        };
        delete formValues.agents_and_groups;
      }

      formValues.max_budget = mapEmptyStringToNull(formValues.max_budget);
      formValues.tpm_limit = mapEmptyStringToNull(formValues.tpm_limit);
      formValues.rpm_limit = mapEmptyStringToNull(formValues.rpm_limit);
      formValues.max_parallel_requests = mapEmptyStringToNull(formValues.max_parallel_requests);

      // Convert metadata back to an object if it exists and is a string
      if (formValues.metadata && typeof formValues.metadata === "string") {
        try {
          const parsedMetadata = JSON.parse(formValues.metadata);
          // Ensure tags are controlled via dedicated field, not in metadata textarea
          if ("tags" in parsedMetadata) {
            delete parsedMetadata["tags"];
          }
          formValues.metadata = {
            ...parsedMetadata,
            ...(Array.isArray(formValues.tags) && formValues.tags.length > 0 ? { tags: formValues.tags } : {}),
            ...(formValues.guardrails?.length > 0 ? { guardrails: formValues.guardrails } : {}),
            ...(Array.isArray(formValues.logging_settings) && formValues.logging_settings.length > 0
              ? { logging: formValues.logging_settings }
              : {}),
            ...(formValues.disabled_callbacks?.length > 0
              ? {
                  litellm_disabled_callbacks: mapDisplayToInternalNames(formValues.disabled_callbacks),
                }
              : {}),
          };
        } catch (error) {
          console.error("Error parsing metadata JSON:", error);
          toast.error("Invalid metadata JSON");
          return;
        }
      } else {
        const baseMetadata = formValues.metadata || {};
        const { tags: _omitTags, ...rest } = baseMetadata;
        formValues.metadata = {
          ...rest,
          ...(Array.isArray(formValues.tags) && formValues.tags.length > 0 ? { tags: formValues.tags } : {}),
          ...(formValues.guardrails?.length > 0 ? { guardrails: formValues.guardrails } : {}),
          ...(Array.isArray(formValues.logging_settings) && formValues.logging_settings.length > 0
            ? { logging: formValues.logging_settings }
            : {}),
          ...(formValues.disabled_callbacks?.length > 0
            ? {
                litellm_disabled_callbacks: mapDisplayToInternalNames(formValues.disabled_callbacks),
              }
            : {}),
        };
      }

      // tags are merged into metadata; do not send as top-level field
      if ("tags" in formValues) {
        delete formValues.tags;
      }
      delete formValues.logging_settings;

      // Normalize any legacy word-form budget_duration to the canonical API format
      if (formValues.budget_duration) {
        const wordToCanonical: Record<string, string> = {
          hourly: "1h",
          daily: "24h",
          weekly: "7d",
          monthly: "30d",
        };
        formValues.budget_duration = wordToCanonical[formValues.budget_duration] ?? formValues.budget_duration;
      }

      const newKeyValues = await keyUpdateCall(accessToken, formValues);

      // Update local state
      setCurrentKeyData((prevData) => (prevData ? { ...prevData, ...newKeyValues } : undefined));

      if (onKeyDataUpdate) {
        onKeyDataUpdate(newKeyValues);
      }
      toast.success("Key updated successfully");
      setIsEditing(false);
      // Refresh key data here if needed
    } catch (error) {
      toast.fromError(parseErrorMessage(error));
      console.error("Error updating key:", error);
    }
  };

  const handleDelete = async () => {
    try {
      setDeleteLoading(true);
      if (!accessToken) return;
      await keyDeleteCall(accessToken as string, currentKeyData.token || currentKeyData.token_id);
      toast.success("Key deleted successfully");
      await queryClient.invalidateQueries({ queryKey: keyKeys.lists() });
      if (onDelete) {
        onDelete();
      }
      onClose();
    } catch (error) {
      console.error("Error deleting the key:", error);
      toast.fromError(error);
    } finally {
      setDeleteLoading(false);
      setIsDeleteModalOpen(false);
    }
  };

  const handleRegenerateKeyUpdate = (updatedKeyData: Partial<KeyResponse>) => {
    const regeneratedAt = new Date();
    // Update local state immediately with ALL the new data
    setCurrentKeyData((prevData) => {
      if (!prevData) return undefined;
      const newData = {
        ...prevData,
        ...updatedKeyData, // This should include the new token (key-id)
        // Update the created_at to show when it was regenerated
        created_at: regeneratedAt.toLocaleString(),
      };
      return newData;
    });

    // Track regeneration timestamp
    setLastRegeneratedAt(regeneratedAt);
    setIsRecentlyRegenerated(true);

    setKeyDataUpdateHeldUntilModalClose({
      ...updatedKeyData,
      created_at: regeneratedAt.toLocaleString(),
    });
  };

  const handleRegenerateModalClose = () => {
    setIsRegenerateModalOpen(false);
    if (keyDataUpdateHeldUntilModalClose) {
      setKeyDataUpdateHeldUntilModalClose(null);
      onKeyDataUpdate?.(keyDataUpdateHeldUntilModalClose);
    }
  };

  // Update the formatTimestamp function to use the desired date format
  const formatTimestamp = (timestamp: string | Date) => {
    const date = new Date(timestamp);
    const dateStr = date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
    const timeStr = date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
    return `${dateStr} at ${timeStr}`;
  };

  const canModifyKey =
    isProxyAdminRole(userRole || "") ||
    (teamsData &&
      isUserTeamAdminForSingleTeam(
        teamsData?.filter((team) => team.team_id === currentKeyData.team_id)[0]?.members_with_roles,
        userID || "",
      )) ||
    (userID === currentKeyData.user_id && userRole !== "Internal Viewer");

  const isKeyAdmin =
    isProxyAdminRole(userRole || "") ||
    Boolean(
      teamsData &&
        isUserTeamAdminForSingleTeam(
          teamsData?.filter((team) => team.team_id === currentKeyData.team_id)[0]?.members_with_roles,
          userID || "",
        ),
    );

  const canResetSpend = isKeyAdmin;
  const canBlockKey = isKeyAdmin;

  const handleResetSpend = () => {
    resetKeySpend(currentKeyData.token || currentKeyData.token_id, {
      onSuccess: () => {
        setCurrentKeyData((prevData) => (prevData ? { ...prevData, spend: 0 } : undefined));
        if (onKeyDataUpdate) {
          onKeyDataUpdate({ spend: 0 });
        }
        toast.success("Key spend reset to $0");
        setIsResetSpendModalOpen(false);
      },
      onError: (error) => {
        toast.fromError(parseErrorMessage(error));
        console.error("Error resetting key spend:", error);
      },
    });
  };

  const isBlocked = currentKeyData.blocked === true;

  const handleToggleBlocked = () => {
    setKeyBlockedState(
      { keyToken: currentKeyData.token || currentKeyData.token_id, blocked: !isBlocked },
      {
        onSuccess: (response) => {
          const blocked = response.blocked === true;
          setCurrentKeyData((prevData) => (prevData ? { ...prevData, blocked } : undefined));
          if (onKeyDataUpdate) {
            onKeyDataUpdate({ blocked });
          }
          toast.success(blocked ? "Key blocked" : "Key unblocked");
          setIsBlockModalOpen(false);
        },
        onError: (error) => {
          toast.fromError(parseErrorMessage(error));
          console.error("Error updating key blocked state:", error);
        },
      },
    );
  };

  const lastConfiguredAt = currentKeyData.settings_updated_at || currentKeyData.created_at;

  const parentTeam = currentKeyData.team_id ? teamsData?.find((team) => team.team_id === currentKeyData.team_id) : null;
  const orgId = currentKeyData.organization_id || currentKeyData.org_id || parentTeam?.organization_id || "";
  const parentOrg = orgId ? organizations?.find((org) => org.organization_id === orgId) : null;

  const hasOwnBudget = currentKeyData.max_budget !== null;
  const budgetDisplay = hasOwnBudget ? `$${formatNumberWithCommas(currentKeyData.max_budget, 2)}` : "Unlimited";
  const inheritedGates = hasOwnBudget ? [] : inheritedBudgetGates(parentTeam, parentOrg);

  return (
    <div className="w-full h-full overflow-y-auto p-4">
      <KeyInfoHeader
        data={{
          keyName: currentKeyData.key_alias || "Virtual Key",
          keyId: currentKeyData.token_id || currentKeyData.token,
          userId: currentKeyData.user_id || "",
          userEmail: currentKeyData.user_email || "",
          userAlias: currentKeyData.user?.user_alias ?? null,
          teamId: currentKeyData.team_id || "",
          teamAlias: parentTeam?.team_alias ?? null,
          orgId,
          orgAlias: parentOrg?.organization_alias ?? null,
          createdBy:
            currentKeyData.created_by_user?.user_alias ||
            currentKeyData.created_by_user?.user_email ||
            currentKeyData.created_by ||
            "",
          createdById: currentKeyData.created_by_user?.user_id || currentKeyData.created_by || "",
          createdAt: currentKeyData.created_at ? formatTimestamp(currentKeyData.created_at) : "",
          lastUpdated: lastConfiguredAt ? formatTimestamp(lastConfiguredAt) : "",
          lastActive: currentKeyData.last_active ? formatTimestamp(currentKeyData.last_active) : "Never",
          expires: currentKeyData.expires ? formatTimestamp(currentKeyData.expires) : "Never",
        }}
        onBack={onClose}
        onRegenerate={() => setIsRegenerateModalOpen(true)}
        onDelete={() => setIsDeleteModalOpen(true)}
        onResetSpend={canResetSpend ? () => setIsResetSpendModalOpen(true) : undefined}
        onToggleBlocked={canBlockKey ? () => setIsBlockModalOpen(true) : undefined}
        isBlocked={isBlocked}
        canModifyKey={canModifyKey}
        backButtonText={backButtonText}
        regenerateDisabled={!premiumUser}
        regenerateTooltip={
          !premiumUser ? "This is a LiteLLM Enterprise feature, and requires a valid key to use." : undefined
        }
      />

      {/* Add RegenerateKeyModal */}
      <RegenerateKeyModal
        selectedToken={currentKeyData}
        visible={isRegenerateModalOpen}
        onClose={handleRegenerateModalClose}
        onKeyUpdate={handleRegenerateKeyUpdate}
      />

      {/* Delete Confirmation Modal */}
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Key"
        alertMessage="This action is irreversible and will immediately revoke access for any applications using this key."
        message="Are you sure you want to delete this Virtual Key?"
        resourceInformationTitle="Key Information"
        resourceInformation={[
          {
            label: "Key Alias",
            value: currentKeyData?.key_alias || "-",
          },
          {
            label: "Key ID",
            value: currentKeyData?.token_id || currentKeyData?.token || "-",
            code: true,
          },
          {
            label: "Team ID",
            value: currentKeyData?.team_id || "-",
            code: true,
          },
          {
            label: "Spend",
            value: currentKeyData?.spend ? `$${formatNumberWithCommas(currentKeyData.spend, 4)}` : "$0.0000",
          },
        ]}
        onCancel={() => {
          setIsDeleteModalOpen(false);
        }}
        onOk={handleDelete}
        confirmLoading={deleteLoading}
        requiredConfirmation={currentKeyData?.key_alias}
      />

      {/* Reset Spend Confirmation Modal */}
      <Dialog open={isResetSpendModalOpen} onOpenChange={(open) => setIsResetSpendModalOpen(open)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reset Key Spend</DialogTitle>
          </DialogHeader>
          <p>
            Reset spend for <strong>{currentKeyData?.key_alias || currentKeyData?.token_id || "this key"}</strong> to{" "}
            <strong>$0</strong>?
          </p>
          <p style={{ color: "#666", fontSize: "0.875rem", marginTop: 8 }}>
            Current spend: <strong>${formatNumberWithCommas(currentKeyData.spend, 4)}</strong>. Spend history is
            preserved in logs. This resets the current period spend counter, the same as an automatic budget reset.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsResetSpendModalOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleResetSpend} disabled={resetSpendLoading}>
              Reset
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isBlockModalOpen} onOpenChange={(open) => setIsBlockModalOpen(open)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{isBlocked ? "Unblock Key" : "Block Key"}</DialogTitle>
          </DialogHeader>
          <p>
            {isBlocked ? "Unblock" : "Block"}{" "}
            <strong>{currentKeyData?.key_alias || currentKeyData?.token_id || "this key"}</strong>?
          </p>
          <p style={{ color: "#666", fontSize: "0.875rem", marginTop: 8 }}>
            {isBlocked
              ? "Requests using this key will be accepted again."
              : "Requests using this key will be rejected with a 401 error until it is unblocked. The key is not deleted and can be unblocked at any time."}
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsBlockModalOpen(false)}>
              Cancel
            </Button>
            <Button
              variant={isBlocked ? "default" : "destructive"}
              onClick={handleToggleBlocked}
              disabled={blockLoading}
            >
              {isBlocked ? "Unblock" : "Block"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Tabs defaultValue="overview">
        <TabsList variant="line" className="mb-4 h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            Overview
          </TabsTrigger>
          <TabsTrigger value="savings" className="flex-none rounded-none px-4 py-2">
            Savings
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex-none rounded-none px-4 py-2">
            Settings
          </TabsTrigger>
        </TabsList>

        <div>
          {/* Overview Panel */}
          <TabsContent value="overview" keepMounted>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card className="block p-6">
                <p className="text-sm">Spend</p>
                <div className="mt-2">
                  <h3 className="text-lg font-medium">${formatNumberWithCommas(currentKeyData.spend, 4)}</h3>
                  <p className="text-sm">
                    of {budgetDisplay}
                    <InheritedBudgetHint gates={inheritedGates} />
                  </p>
                  {currentKeyData.budget_reset_at && (
                    <p className="text-sm">Resets {formatTimestamp(currentKeyData.budget_reset_at)}</p>
                  )}
                </div>
              </Card>

              <Card className="block p-6">
                <p className="text-sm">Rate Limits</p>
                <div className="mt-2">
                  <p className="text-sm">
                    TPM: {currentKeyData.tpm_limit !== null ? currentKeyData.tpm_limit : "Unlimited"}
                  </p>
                  <p className="text-sm">
                    RPM: {currentKeyData.rpm_limit !== null ? currentKeyData.rpm_limit : "Unlimited"}
                  </p>
                  {Boolean(currentKeyData.metadata?.throttle_on_budget_exceeded) && (
                    <p className="text-sm">Throttle on budget exceeded: Yes</p>
                  )}
                </div>
              </Card>

              <Card className="block p-6">
                <p className="text-sm">Models</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {currentKeyData.models && currentKeyData.models.length > 0 ? (
                    currentKeyData.models.map((model, index) => (
                      <Badge key={index} variant="secondary" className="min-w-0 break-words">
                        {model}
                      </Badge>
                    ))
                  ) : (
                    <p className="text-sm">No models specified</p>
                  )}
                </div>
              </Card>

              <Card className="block p-6">
                <ObjectPermissionsView
                  objectPermission={currentKeyData.object_permission}
                  variant="inline"
                  accessToken={accessToken}
                />
              </Card>

              <Card className="block p-6">
                <p className="text-sm font-medium mb-3">Guardrails</p>
                {Array.isArray(currentKeyData.metadata?.guardrails) && currentKeyData.metadata.guardrails.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {currentKeyData.metadata.guardrails.map((guardrail: string, index: number) => (
                      <Badge key={index} variant="secondary" className="min-w-0 break-words">
                        {guardrail}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No guardrails configured</p>
                )}
                {typeof currentKeyData.metadata?.disable_global_guardrails === "boolean" &&
                  currentKeyData.metadata.disable_global_guardrails === true && (
                    <div className="mt-3 pt-3 border-t border-border">
                      <Badge variant="destructive">Global Guardrails Disabled</Badge>
                    </div>
                  )}
              </Card>

              <Card className="block p-6">
                <p className="text-sm font-medium mb-3">Policies</p>
                {Array.isArray(currentKeyData.metadata?.policies) && currentKeyData.metadata.policies.length > 0 ? (
                  <div className="space-y-4">
                    {currentKeyData.metadata.policies.map((policy: string, index: number) => (
                      <div key={index} className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary" className="min-w-0 break-words">
                            {policy}
                          </Badge>
                          {loadingPolicies && <p className="text-xs text-muted-foreground">Loading guardrails...</p>}
                        </div>
                        {!loadingPolicies && policyGuardrails[policy] && policyGuardrails[policy].length > 0 && (
                          <div className="ml-4 pl-3 border-l-2 border-border">
                            <p className="text-xs text-muted-foreground mb-1">Resolved Guardrails:</p>
                            <div className="flex flex-wrap gap-1">
                              {policyGuardrails[policy].map((guardrail: string, gIndex: number) => (
                                <Badge key={gIndex} variant="secondary" className="min-w-0 break-words">
                                  {guardrail}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No policies configured</p>
                )}
              </Card>

              <LoggingSettingsView
                loggingConfigs={extractLoggingSettings(currentKeyData.metadata)}
                disabledCallbacks={
                  Array.isArray(currentKeyData.metadata?.litellm_disabled_callbacks)
                    ? mapInternalToDisplayNames(currentKeyData.metadata.litellm_disabled_callbacks)
                    : []
                }
                variant="card"
              />

              <AutoRotationView
                autoRotate={currentKeyData.auto_rotate}
                rotationInterval={currentKeyData.rotation_interval}
                lastRotationAt={currentKeyData.last_rotation_at}
                keyRotationAt={currentKeyData.key_rotation_at}
                nextRotationAt={currentKeyData.next_rotation_at}
                variant="card"
              />
            </div>
          </TabsContent>

          {/* Savings Panel. No keepMounted: this tab sweeps the daily rollup, and staying mounted
              would fire that request on every key page open for people who never look at it. */}
          <TabsContent value="savings">
            <KeySavingsTab
              accessToken={accessToken}
              keyToken={currentKeyData.token}
              userId={userID}
              userRole={userRole}
            />
          </TabsContent>

          {/* Settings Panel */}
          <TabsContent value="settings" keepMounted>
            <Card className="block p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">Key Settings</h3>
                {!isEditing && canModifyKey && (
                  <Button variant="outline" onClick={() => setIsEditing(true)}>
                    Edit Settings
                  </Button>
                )}
              </div>

              {isEditing ? (
                <KeyEditView
                  keyData={currentKeyData}
                  onCancel={() => setIsEditing(false)}
                  onSubmit={handleKeyUpdate}
                  teams={teams}
                  accessToken={accessToken}
                  userID={userID}
                  userRole={userRole}
                  premiumUser={premiumUser}
                />
              ) : (
                <div className="space-y-4">
                  <div>
                    <p className="text-sm font-medium">Key ID</p>
                    <p className="text-sm font-mono">{currentKeyData.token_id || currentKeyData.token}</p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Key Alias</p>
                    <p className="text-sm">{currentKeyData.key_alias || "Not Set"}</p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Secret Key</p>
                    <p className="text-sm font-mono">{currentKeyData.key_name}</p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Team ID</p>
                    <p className="text-sm">
                      {currentKeyData.team_id ? (
                        <EntityLink href={teamDetailHref(currentKeyData.team_id)} className="font-normal">
                          {currentKeyData.team_id}
                        </EntityLink>
                      ) : (
                        "Not Set"
                      )}
                    </p>
                  </div>

                  {enableProjectsUI && (
                    <div>
                      <p className="text-sm font-medium">Project</p>
                      <p className="text-sm">
                        {currentKeyData.project_id
                          ? (() => {
                              const project = projects?.find((p) => p.project_id === currentKeyData.project_id);
                              return project?.project_alias
                                ? `${project.project_alias} (${currentKeyData.project_id})`
                                : currentKeyData.project_id;
                            })()
                          : "Not Set"}
                      </p>
                    </div>
                  )}

                  <div>
                    <p className="text-sm font-medium">Organization</p>
                    <p className="text-sm">{(currentKeyData.organization_id ?? currentKeyData.org_id) || "Not Set"}</p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Created</p>
                    <p className="text-sm">{formatTimestamp(currentKeyData.created_at)}</p>
                  </div>

                  {lastRegeneratedAt && (
                    <div>
                      <p className="text-sm font-medium">Last Regenerated</p>
                      <div className="flex items-center gap-2">
                        <p className="text-sm">{formatTimestamp(lastRegeneratedAt)}</p>
                        <Badge variant="secondary">Recent</Badge>
                      </div>
                    </div>
                  )}

                  <div>
                    <p className="text-sm font-medium">Expires</p>
                    <p className="text-sm">
                      {currentKeyData.expires ? formatTimestamp(currentKeyData.expires) : "Never"}
                    </p>
                  </div>

                  {Boolean(currentKeyData.metadata?.enable_prompt_caching) && (
                    <div>
                      <p className="text-sm font-medium">Prompt Caching</p>
                      <p className="text-sm">
                        Enabled (auto-injects cache_control markers on Anthropic and Bedrock Claude requests)
                      </p>
                    </div>
                  )}

                  <AutoRotationView
                    autoRotate={currentKeyData.auto_rotate}
                    rotationInterval={currentKeyData.rotation_interval}
                    lastRotationAt={currentKeyData.last_rotation_at}
                    keyRotationAt={currentKeyData.key_rotation_at}
                    nextRotationAt={currentKeyData.next_rotation_at}
                    variant="inline"
                    className="pt-4 border-t border-border"
                  />

                  <div>
                    <p className="text-sm font-medium">Spend</p>
                    <p className="text-sm">${formatNumberWithCommas(currentKeyData.spend, 4)} USD</p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Budget</p>
                    <p className="text-sm">
                      {currentKeyData.max_budget !== null
                        ? `$${formatNumberWithCommas(currentKeyData.max_budget, 2)}`
                        : "Unlimited"}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Budget Reset</p>
                    <p className="text-sm">
                      {currentKeyData.budget_reset_at
                        ? `${currentKeyData.budget_duration ? `Every ${currentKeyData.budget_duration}, next ` : ""}${formatTimestamp(currentKeyData.budget_reset_at)}`
                        : "Never"}
                    </p>
                  </div>

                  {currentKeyData.budget_fallbacks && Object.keys(currentKeyData.budget_fallbacks).length > 0 && (
                    <div>
                      <p className="text-sm font-medium">Budget Fallbacks</p>
                      <div className="mt-1 space-y-1">
                        {Object.entries(currentKeyData.budget_fallbacks).map(([model, fallbacks]) => (
                          <div key={model} className="text-xs text-muted-foreground">
                            <span className="font-medium">{model}</span>
                            <span className="mx-1 text-muted-foreground">-&gt;</span>
                            {fallbacks.join(", ")}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {hasRouterSettings(currentKeyData.router_settings) && (
                    <div>
                      <p className="text-sm font-medium">Router Settings</p>
                      <div className="mt-1">
                        <RouterSettingsSummary routerSettings={currentKeyData.router_settings} />
                      </div>
                    </div>
                  )}

                  <div>
                    <p className="text-sm font-medium">Tags</p>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {Array.isArray(currentKeyData.metadata?.tags) && currentKeyData.metadata.tags.length > 0
                        ? currentKeyData.metadata.tags.map((tag, index) => (
                            <span key={index} className="px-2 mr-2 py-1 bg-info/15 rounded-sm text-xs">
                              {tag}
                            </span>
                          ))
                        : "No tags specified"}
                    </div>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Prompts</p>
                    <p className="text-sm">
                      {Array.isArray(currentKeyData.metadata?.prompts) && currentKeyData.metadata.prompts.length > 0
                        ? currentKeyData.metadata.prompts.map((prompt, index) => (
                            <span key={index} className="px-2 mr-2 py-1 bg-info/15 rounded-sm text-xs">
                              {prompt}
                            </span>
                          ))
                        : "No prompts specified"}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Allowed Routes</p>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {Array.isArray(currentKeyData.allowed_routes) && currentKeyData.allowed_routes.length > 0 ? (
                        currentKeyData.allowed_routes.map((route, index) => (
                          <span key={index} className="px-2 py-1 bg-info/15 rounded-sm text-xs">
                            {route}
                          </span>
                        ))
                      ) : (
                        <Badge variant="secondary">All routes allowed</Badge>
                      )}
                    </div>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Allowed Pass Through Routes</p>
                    <p className="text-sm">
                      {Array.isArray(currentKeyData.metadata?.allowed_passthrough_routes) &&
                      currentKeyData.metadata.allowed_passthrough_routes.length > 0
                        ? currentKeyData.metadata.allowed_passthrough_routes.map((route, index) => (
                            <span key={index} className="px-2 mr-2 py-1 bg-info/15 rounded-sm text-xs">
                              {route}
                            </span>
                          ))
                        : "No pass through routes specified"}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Disable Global Guardrails</p>
                    <p className="text-sm">
                      {currentKeyData.metadata?.disable_global_guardrails === true ? (
                        <Badge variant="destructive">Enabled - Global guardrails bypassed</Badge>
                      ) : (
                        <Badge variant="secondary">Disabled - Global guardrails active</Badge>
                      )}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Models</p>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {currentKeyData.models && currentKeyData.models.length > 0 ? (
                        currentKeyData.models.map((model, index) => (
                          <span key={index} className="px-2 py-1 bg-info/15 rounded-sm text-xs">
                            {model}
                          </span>
                        ))
                      ) : (
                        <p className="text-sm">No models specified</p>
                      )}
                    </div>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Rate Limits</p>
                    <p className="text-sm">
                      TPM: {currentKeyData.tpm_limit !== null ? currentKeyData.tpm_limit : "Unlimited"}
                    </p>
                    <p className="text-sm">
                      RPM: {currentKeyData.rpm_limit !== null ? currentKeyData.rpm_limit : "Unlimited"}
                    </p>
                    <p className="text-sm">
                      Max Parallel Requests:{" "}
                      {currentKeyData.max_parallel_requests !== null
                        ? currentKeyData.max_parallel_requests
                        : "Unlimited"}
                    </p>
                    <p className="text-sm">
                      Model TPM Limits:{" "}
                      {currentKeyData.metadata?.model_tpm_limit
                        ? JSON.stringify(currentKeyData.metadata.model_tpm_limit)
                        : "Unlimited"}
                    </p>
                    <p className="text-sm">
                      Model RPM Limits:{" "}
                      {currentKeyData.metadata?.model_rpm_limit
                        ? JSON.stringify(currentKeyData.metadata.model_rpm_limit)
                        : "Unlimited"}
                    </p>
                    <p className="text-sm">
                      Tag RPM Limits:{" "}
                      {currentKeyData.metadata?.tag_rpm_limit &&
                      Object.keys(currentKeyData.metadata.tag_rpm_limit).length > 0
                        ? JSON.stringify(currentKeyData.metadata.tag_rpm_limit)
                        : "Unlimited"}
                    </p>
                    <p className="text-sm">
                      Estimated Output Tokens:{" "}
                      {currentKeyData.metadata?.default_estimated_output_tokens != null
                        ? String(currentKeyData.metadata.default_estimated_output_tokens)
                        : "Default"}
                    </p>
                    <p className="text-sm">
                      Estimated Output Tokens Per Model:{" "}
                      {currentKeyData.metadata?.default_estimated_output_tokens_per_model
                        ? JSON.stringify(currentKeyData.metadata.default_estimated_output_tokens_per_model)
                        : "Default"}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Metadata</p>
                    <pre className="bg-muted p-2 rounded-sm text-xs overflow-auto mt-1">
                      {formatMetadataForDisplay(stripTagsFromMetadata(currentKeyData.metadata))}
                    </pre>
                  </div>

                  <ObjectPermissionsView
                    objectPermission={currentKeyData.object_permission}
                    variant="inline"
                    className="pt-4 border-t border-border"
                    accessToken={accessToken}
                  />

                  <LoggingSettingsView
                    loggingConfigs={extractLoggingSettings(currentKeyData.metadata)}
                    disabledCallbacks={
                      Array.isArray(currentKeyData.metadata?.litellm_disabled_callbacks)
                        ? mapInternalToDisplayNames(currentKeyData.metadata.litellm_disabled_callbacks)
                        : []
                    }
                    variant="inline"
                    className="pt-4 border-t border-border"
                  />
                </div>
              )}
            </Card>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
