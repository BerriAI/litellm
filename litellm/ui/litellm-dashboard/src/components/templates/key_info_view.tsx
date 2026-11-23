import { LensControls } from "@/components/lens/LensControls";
import { readLensPolicy, type LensPolicy } from "@/components/lens/lensPolicy";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { lensEntityHref } from "@/components/lens/entityLinks";
import { useUrlTab } from "@/hooks/useUrlTab";
import ResourceActivity from "@/components/UsagePage/ResourceActivity";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelAccessGroupNames } from "@/app/(dashboard)/hooks/models/useModels";
import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useApplyUserBudgetToTeamKeys } from "@/app/(dashboard)/hooks/uiSettings/useApplyUserBudgetToTeamKeys";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { formatBudgetResetTime } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { mapEmptyStringToNull } from "@/utils/keyUpdateUtils";
import { canModifyKey as canModifyVirtualKey } from "@/utils/keyPermissions";
import { ArrowLeft, SearchX } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { EntityLink } from "@/components/shared/EntityLink";
import { modelOrAccessGroupHref, orgDetailHref, teamDetailHref, userDetailHref } from "@/utils/entityLinks";
import { BadgeLink } from "@/components/shared/BadgeLink";
import { KeyInfoHeader, type KeyStatus } from "./KeyInfoHeader";
import { usedUpBudget } from "./key_detail/canStillSpend";
import { KeyOverview } from "./key_detail/KeyOverview";
import { KeyTabsList } from "./key_detail/KeyTabsList";
import KeySavingsTab from "./KeySavingsTab";
import KeyAutoRouterUsageTab from "./KeyAutoRouterUsageTab";
import { useActivityDateRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";
import { useEffect, useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import {
  hasProxyWideSpendView,
  isProxyAdminRole,
  isUserTeamAdminForSingleTeam,
  rolesWithWriteAccess,
} from "../../utils/roles";
import { mapDisplayToInternalNames, mapInternalToDisplayNames } from "../callback_info_helpers";
import AutoRotationView from "../common_components/AutoRotationView";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import RouterSettingsSummary from "../common_components/RouterSettingsSummary";
import { hasRouterSettings } from "../common_components/routerSettingsPayload";
import { extractLoggingSettings, formatMetadataForDisplay, stripTagsFromMetadata } from "../key_info_utils";
import {
  budgetAlertEmailsFromRows,
  budgetAlertEmailsMetadata,
  budgetAlertEmailsRowsCanShow,
  KEY_MAX_BUDGET_ALERT_EMAILS_KEY,
} from "../key_team_helpers/budgetAlertEmails";
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
import { useTeamMemberBudgets } from "@/app/(dashboard)/hooks/teams/useTeamMemberBudgets";
import { extractMcpEntitlement } from "../mcp_server_management/mcpEntitlement";
import ObjectPermissionsView from "../object_permissions_view";
import { RegenerateKeyModal } from "../organisms/RegenerateKeyModal";
import { parseErrorMessage } from "../shared/errorUtils";
import { appliedBudgets, type AppliedBudget } from "../shared/appliedBudgets";
import { inheritedBudgetGates, keyOwnerBudgetSource, teamMemberBudgetGate } from "../shared/InheritedBudgetHint";
import { KeyEditView } from "./key_edit_view";
import { keyOrganizationId } from "./keyEditFieldNormalizers";
import { isTeamAdminEditingMemberKey, teamAdminMemberKeyPayload } from "./teamAdminMemberKeyPayload";

export function keyStatusOf(state: { blocked: boolean; overBudget: boolean; expired: boolean }): KeyStatus {
  if (state.blocked) return "blocked";
  if (state.overBudget) return "overBudget";
  return state.expired ? "expired" : "active";
}

export function needsLifetimeSpendBackfill(spend: number, totalSpend: number | null | undefined): boolean {
  return (totalSpend ?? 0) < spend;
}

interface KeyInfoViewProps {
  keyId: string;
  onClose: () => void;
  keyData: KeyResponse | undefined;
  onKeyDataUpdate?: (data: Partial<KeyResponse>) => void;
  onDelete?: () => void;
  teams: any[] | null;
  backButtonText?: string;
  tabParam?: string;
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

export default function KeyInfoView({
  onClose,
  keyData,
  teams,
  onKeyDataUpdate,
  onDelete,
  backButtonText = i18n.t("keyDetail.keyInfoView.backToKeys"),
  tabParam = "detail_tab",
}: KeyInfoViewProps) {
  const { t, i18n: locale } = useTranslation();
  const { accessToken, userId: userID, userRole, premiumUser, isViewOnly } = useAuthorized();
  const accessGroupNames = useModelAccessGroupNames();
  const activityDateRange = useActivityDateRange();
  const [activeTab, setActiveTab] = useUrlTab(
    ["overview", "activity", "lens", "savings", "auto-router-usage", "settings"],
    "overview",
    tabParam,
  );
  const queryClient = useQueryClient();
  const canEditGuardrails = premiumUser || (userRole != null && rolesWithWriteAccess.includes(userRole));
  const { teams: teamsData } = useTeams();
  const { data: organizations } = useOrganizations();
  const { data: projects } = useProjects();
  const applyUserBudgetToTeamKeys = useApplyUserBudgetToTeamKeys();
  const { data: allMcpServers } = useMCPServers();
  const { data: allMcpToolsets } = useMCPToolsets();
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
  const lensTeam = useTeam(activeTab === "lens" ? currentKeyData?.team_id ?? undefined : undefined);
  const memberBudgetUserIds = useMemo(
    (): Record<string, string[]> =>
      currentKeyData?.team_id && currentKeyData.user_id ? { [currentKeyData.team_id]: [currentKeyData.user_id] } : {},
    [currentKeyData],
  );
  const teamMemberBudgets = useTeamMemberBudgets(memberBudgetUserIds);
  const [lastRegeneratedAt, setLastRegeneratedAt] = useState<Date | null>(null);
  const [keyDataUpdateHeldUntilModalClose, setKeyDataUpdateHeldUntilModalClose] = useState<Partial<KeyResponse> | null>(
    null,
  );
  const [isRecentlyRegenerated, setIsRecentlyRegenerated] = useState(false);
  const [policyGuardrails, setPolicyGuardrails] = useState<Record<string, string[]>>({});
  const [loadingPolicies, setLoadingPolicies] = useState(false);
  const [nowMs] = useState(() => Date.now());

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
      <div className="h-full w-full overflow-y-auto p-4 sm:p-6">
        <div className="mx-auto max-w-[1180px]">
          <button
            type="button"
            onClick={onClose}
            className="mb-2 inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="size-3.5" />
            {backButtonText}
          </button>
          <Empty className="mt-6">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <SearchX />
              </EmptyMedia>
              <EmptyTitle>{t("keyDetail.keyInfoView.keyNotFound")}</EmptyTitle>
              <EmptyDescription>{t("keyDetail.keyInfoView.keyNotFoundDescription")}</EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <Button variant="outline" onClick={onClose}>
                {backButtonText}
              </Button>
            </EmptyContent>
          </Empty>
        </div>
      </div>
    );
  }

  const memberKeyEditContext = {
    userRole: userRole || "",
    userId: userID || "",
    keyUserId: currentKeyData.user_id,
    keyTeamId: currentKeyData.team_id,
    teamMembers: teamsData?.find((team) => team.team_id === currentKeyData.team_id)?.members_with_roles,
  };
  const editingMemberKeyAsTeamAdmin = isTeamAdminEditingMemberKey(memberKeyEditContext);

  const handleKeyUpdate = async (formValues: Record<string, any>, dirtyFields: readonly string[] = []) => {
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

      if ((formValues.organization_id || null) === keyOrganizationId(currentKeyData)) {
        delete formValues.organization_id;
      }

      // Handle max budget empty string
      formValues.max_budget = mapEmptyStringToNull(formValues.max_budget);

      // soft_budget is a budget change server-side (admin-gated); only send it when it changed
      // so a non-admin edit of unrelated fields isn't blocked by that gate.
      const previousSoftBudget =
        (currentKeyData.litellm_budget_table as { soft_budget?: number | null } | null | undefined)?.soft_budget ??
        null;
      const nextSoftBudget =
        formValues.soft_budget === "" || formValues.soft_budget == null ? null : Number(formValues.soft_budget);
      if (nextSoftBudget !== null && !Number.isFinite(nextSoftBudget)) {
        toast.error(t("keyDetail.keyInfoView.toasts.softBudgetNotFinite"));
        return;
      }
      if (nextSoftBudget === previousSoftBudget) {
        delete formValues.soft_budget;
      } else {
        formValues.soft_budget = nextSoftBudget;
      }

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
          toast.error(t("keyDetail.keyInfoView.toasts.mcpListUnavailable"));
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

      if (formValues.skills !== undefined) {
        formValues.object_permission = {
          ...formValues.object_permission,
          skills: formValues.skills || [],
        };
        delete formValues.skills;
      }

      formValues.max_budget = mapEmptyStringToNull(formValues.max_budget);
      formValues.tpm_limit = mapEmptyStringToNull(formValues.tpm_limit);
      formValues.rpm_limit = mapEmptyStringToNull(formValues.rpm_limit);
      formValues.tpd_limit = mapEmptyStringToNull(formValues.tpd_limit);
      formValues.max_parallel_requests = mapEmptyStringToNull(formValues.max_parallel_requests);

      const budgetAlertRowEntries = dirtyFields.includes(KEY_MAX_BUDGET_ALERT_EMAILS_KEY)
        ? budgetAlertEmailsFromRows(formValues.max_budget_alert_emails ?? [])
        : budgetAlertEmailsRowsCanShow(currentKeyData.metadata, KEY_MAX_BUDGET_ALERT_EMAILS_KEY);
      const budgetAlertEmailsWithBoxEntries = (boxMetadata: unknown) =>
        budgetAlertEmailsMetadata(budgetAlertRowEntries, KEY_MAX_BUDGET_ALERT_EMAILS_KEY, boxMetadata);

      // Convert metadata back to an object if it exists and is a string
      if (formValues.metadata && typeof formValues.metadata === "string") {
        try {
          const parsedMetadata = JSON.parse(formValues.metadata);
          // Ensure tags are controlled via dedicated field, not in metadata textarea
          if ("tags" in parsedMetadata) {
            delete parsedMetadata["tags"];
          }
          const budgetAlertEmails = budgetAlertEmailsWithBoxEntries(parsedMetadata);
          delete parsedMetadata[KEY_MAX_BUDGET_ALERT_EMAILS_KEY];
          formValues.metadata = {
            ...parsedMetadata,
            ...(Array.isArray(formValues.tags) && formValues.tags.length > 0 ? { tags: formValues.tags } : {}),
            ...budgetAlertEmails,
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
          toast.error(t("keyDetail.keyInfoView.toasts.invalidMetadataJson"));
          return;
        }
      } else {
        const baseMetadata = formValues.metadata || {};
        const { tags: _omitTags, [KEY_MAX_BUDGET_ALERT_EMAILS_KEY]: _boxAlertEmails, ...rest } = baseMetadata;
        formValues.metadata = {
          ...rest,
          ...(Array.isArray(formValues.tags) && formValues.tags.length > 0 ? { tags: formValues.tags } : {}),
          ...budgetAlertEmailsWithBoxEntries(baseMetadata),
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
      delete formValues.max_budget_alert_emails;

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

      if (editingMemberKeyAsTeamAdmin) {
        const trimmed = teamAdminMemberKeyPayload(formValues, dirtyFields);
        if (trimmed.kind === "blocked") {
          toast.error(
            t("keyDetail.keyInfoView.toasts.teamAdminBudgetFieldsOnly", { fields: trimmed.fields.join(", ") }),
          );
          return;
        }
        formValues = trimmed.payload;
      }

      const newKeyValues = await keyUpdateCall(accessToken, formValues);

      setCurrentKeyData((prevData) =>
        prevData ? { ...prevData, ...newKeyValues, org_id: newKeyValues.organization_id } : undefined,
      );
      void queryClient.invalidateQueries({ queryKey: keyKeys.all });

      if (onKeyDataUpdate) {
        onKeyDataUpdate(newKeyValues);
      }
      toast.success(t("keyDetail.keyInfoView.toasts.updated"));
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
      toast.success(t("keyDetail.keyInfoView.toasts.deleted"));
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
        created_at: regeneratedAt.toISOString(),
      };
      return newData;
    });

    // Track regeneration timestamp
    setLastRegeneratedAt(regeneratedAt);
    setIsRecentlyRegenerated(true);

    setKeyDataUpdateHeldUntilModalClose({
      ...updatedKeyData,
      created_at: regeneratedAt.toISOString(),
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
    const dateStr = date.toLocaleDateString(locale.language, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
    const timeStr = date.toLocaleTimeString(locale.language, {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
    return t("keyDetail.keyInfoView.timestamp", { date: dateStr, time: timeStr });
  };

  const canModifyKey = canModifyVirtualKey(currentKeyData, teamsData, userID, userRole);
  const canEditLens = canModifyKey && !isViewOnly && !editingMemberKeyAsTeamAdmin;
  const lensTeamStatus = currentKeyData.team_id && lensTeam.isPending ? "loading" : "ready";
  const inheritedLensStatus = lensTeam.isError ? "error" : lensTeamStatus;

  const updateLensPolicy = async (policy: LensPolicy): Promise<void> => {
    if (!accessToken || !canEditLens) return;
    const metadata = { ...currentKeyData.metadata, lens_enabled: policy };
    await keyUpdateCall(accessToken, { key: currentKeyData.token, lens_enabled: policy });
    setCurrentKeyData((current) => (current ? { ...current, metadata } : current));
    onKeyDataUpdate?.({ metadata });
    void queryClient.invalidateQueries({ queryKey: keyKeys.all });
  };

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
        toast.success(t("keyDetail.keyInfoView.toasts.spendReset"));
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
          toast.success(
            blocked ? t("keyDetail.keyInfoView.toasts.blocked") : t("keyDetail.keyInfoView.toasts.unblocked"),
          );
          setIsBlockModalOpen(false);
        },
        onError: (error) => {
          toast.fromError(parseErrorMessage(error));
          console.error("Error updating key blocked state:", error);
        },
      },
    );
  };

  const parentTeam = currentKeyData.team_id ? teamsData?.find((team) => team.team_id === currentKeyData.team_id) : null;
  const orgId = keyOrganizationId(currentKeyData) || parentTeam?.organization_id || "";
  const parentOrg = orgId ? organizations?.find((org) => org.organization_id === orgId) : null;

  const hasOwnBudget = currentKeyData.max_budget !== null;
  const ownerUser = keyOwnerBudgetSource(currentKeyData, applyUserBudgetToTeamKeys);
  const memberGate = teamMemberBudgetGate(
    teamMemberBudgets[currentKeyData.team_id ?? ""],
    currentKeyData.user_id,
    currentKeyData.user?.user_email,
  );
  const budgetDisplay = (() => {
    if (hasOwnBudget) return `$${formatNumberWithCommas(currentKeyData.max_budget, 2)}`;
    if (memberGate)
      return t("keyDetail.keyInfoView.budget.teamMemberBudget", {
        budget: formatNumberWithCommas(memberGate.maxBudget, 2),
      });
    return t("keyDetail.keyInfoView.budget.unlimited");
  })();
  const budgetsThatApply = appliedBudgets(
    currentKeyData,
    currentKeyData.key_alias || t("shared.appliedBudgets.thisKey"),
    inheritedBudgetGates(parentTeam, parentOrg, ownerUser, memberGate),
  );
  const budgetResetTime = formatBudgetResetTime(currentKeyData.budget_reset_at, t, locale.language);

  const keyStatus = keyStatusOf({
    blocked: isBlocked,
    overBudget: usedUpBudget(budgetsThatApply, currentKeyData.metadata?.throttle_on_budget_exceeded === true) !== null,
    expired: Boolean(currentKeyData.expires) && Date.parse(currentKeyData.expires) <= nowMs,
  });
  const project = currentKeyData.project_id
    ? projects?.find((candidate) => candidate.project_id === currentKeyData.project_id)
    : undefined;
  const budgetHref = (budget: AppliedBudget): string | undefined => {
    switch (budget.scope) {
      case "Team":
      case "Team member":
        return currentKeyData.team_id ? teamDetailHref(currentKeyData.team_id) : undefined;
      case "Organization":
        return orgId ? orgDetailHref(orgId) : undefined;
      case "User":
        return ownerUser?.user_id ? userDetailHref(ownerUser.user_id) : undefined;
      case "Key":
        return undefined;
    }
  };
  const createdById = currentKeyData.created_by_user?.user_id || currentKeyData.created_by || "";
  const createdByLabel =
    currentKeyData.created_by_user?.user_alias ||
    currentKeyData.created_by_user?.user_email ||
    currentKeyData.created_by ||
    "";

  return (
    <div className="h-full w-full overflow-y-auto p-4 sm:p-6">
      <div className="mx-auto flex max-w-[1180px] flex-col gap-6">
        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(String(value) as typeof activeTab)}
          className="gap-6"
        >
          <KeyInfoHeader
            data={{
              keyName: currentKeyData.key_alias || t("keyDetail.keyInfoView.header.defaultKeyName"),
              keyId: currentKeyData.token_id || currentKeyData.token,
              secretKey: currentKeyData.key_name || "",
              userId: currentKeyData.user_id || "",
              userEmail: currentKeyData.user_email || "",
              userAlias: currentKeyData.user?.user_alias ?? null,
              teamId: currentKeyData.team_id || "",
              teamAlias: parentTeam?.team_alias ?? null,
              orgId,
              orgAlias: parentOrg?.organization_alias ?? null,
              projectId: currentKeyData.project_id,
              projectAlias: project?.project_alias ?? null,
            }}
            status={keyStatus}
            onBack={onClose}
            onRegenerate={() => setIsRegenerateModalOpen(true)}
            onDelete={() => setIsDeleteModalOpen(true)}
            onResetSpend={canResetSpend ? () => setIsResetSpendModalOpen(true) : undefined}
            onToggleBlocked={canBlockKey ? () => setIsBlockModalOpen(true) : undefined}
            isBlocked={isBlocked}
            canModifyKey={canModifyKey}
            backButtonText={backButtonText}
            regenerateDisabled={!premiumUser}
            regenerateTooltip={!premiumUser ? t("keyDetail.keyInfoView.header.enterpriseOnly") : undefined}
            tabs={<KeyTabsList showAutoRouter={hasProxyWideSpendView(userRole)} />}
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
            title={t("keyDetail.keyInfoView.deleteModal.title")}
            alertMessage={t("keyDetail.keyInfoView.deleteModal.alert")}
            message={t("keyDetail.keyInfoView.deleteModal.message")}
            resourceInformationTitle={t("keyDetail.keyInfoView.deleteModal.informationTitle")}
            resourceInformation={[
              {
                label: t("keyDetail.keyInfoView.deleteModal.keyAlias"),
                value: currentKeyData?.key_alias || "-",
              },
              {
                label: t("keyDetail.keyInfoView.deleteModal.keyId"),
                value: currentKeyData?.token_id || currentKeyData?.token || "-",
                code: true,
              },
              {
                label: t("keyDetail.keyInfoView.deleteModal.teamId"),
                value: currentKeyData?.team_id || "-",
                code: true,
              },
              {
                label: t("keyDetail.keyInfoView.deleteModal.spend"),
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
                <DialogTitle>{t("keyDetail.keyInfoView.resetSpendDialog.title")}</DialogTitle>
              </DialogHeader>
              <p>
                <Trans
                  i18nKey="keyDetail.keyInfoView.resetSpendDialog.prompt"
                  components={{
                    strong: <strong />,
                    alias: (
                      <>
                        {currentKeyData?.key_alias ||
                          currentKeyData?.token_id ||
                          t("keyDetail.keyInfoView.resetSpendDialog.thisKey")}
                      </>
                    ),
                  }}
                />
              </p>
              <p style={{ color: "#666", fontSize: "0.875rem", marginTop: 8 }}>
                <Trans
                  i18nKey="keyDetail.keyInfoView.resetSpendDialog.summary"
                  components={{ strong: <strong />, spend: <>{formatNumberWithCommas(currentKeyData.spend, 4)}</> }}
                />
              </p>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsResetSpendModalOpen(false)}>
                  {t("keyDetail.keyInfoView.resetSpendDialog.cancel")}
                </Button>
                <Button variant="destructive" onClick={handleResetSpend} disabled={resetSpendLoading}>
                  {t("keyDetail.keyInfoView.resetSpendDialog.confirm")}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Dialog open={isBlockModalOpen} onOpenChange={(open) => setIsBlockModalOpen(open)}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>
                  {isBlocked
                    ? t("keyDetail.keyInfoView.blockDialog.unblockTitle")
                    : t("keyDetail.keyInfoView.blockDialog.blockTitle")}
                </DialogTitle>
              </DialogHeader>
              <p>
                {isBlocked
                  ? t("keyDetail.keyInfoView.blockDialog.unblock")
                  : t("keyDetail.keyInfoView.blockDialog.block")}{" "}
                <strong>
                  {currentKeyData?.key_alias ||
                    currentKeyData?.token_id ||
                    t("keyDetail.keyInfoView.blockDialog.thisKey")}
                </strong>
                ?
              </p>
              <p style={{ color: "#666", fontSize: "0.875rem", marginTop: 8 }}>
                {isBlocked
                  ? t("keyDetail.keyInfoView.blockDialog.unblockDescription")
                  : t("keyDetail.keyInfoView.blockDialog.blockDescription")}
              </p>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsBlockModalOpen(false)}>
                  {t("keyDetail.keyInfoView.blockDialog.cancel")}
                </Button>
                <Button
                  variant={isBlocked ? "default" : "destructive"}
                  onClick={handleToggleBlocked}
                  disabled={blockLoading}
                >
                  {isBlocked
                    ? t("keyDetail.keyInfoView.blockDialog.confirmUnblock")
                    : t("keyDetail.keyInfoView.blockDialog.confirmBlock")}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <div>
            <TabsContent value="overview" keepMounted>
              <KeyOverview
                keyData={currentKeyData}
                budgets={budgetsThatApply}
                budgetHref={budgetHref}
                team={
                  currentKeyData.team_id
                    ? { id: currentKeyData.team_id, name: parentTeam?.team_alias || currentKeyData.team_id }
                    : null
                }
                createdBy={createdByLabel || createdById ? { label: createdByLabel, userId: createdById } : null}
                accessGroupNames={accessGroupNames}
                policyGuardrails={policyGuardrails}
                loadingPolicies={loadingPolicies}
                lifetimeNeedsBackfill={needsLifetimeSpendBackfill(currentKeyData.spend, currentKeyData.total_spend)}
                onOpenActivity={() => setActiveTab("activity")}
                onUnblock={canBlockKey ? () => setIsBlockModalOpen(true) : undefined}
                nowMs={nowMs}
              />
            </TabsContent>

            <TabsContent value="activity">
              <ResourceActivity scope="key" id={currentKeyData.token} activity={activityDateRange} />
            </TabsContent>

            <TabsContent value="lens">
              <LensControls
                kind="key"
                policy={readLensPolicy(currentKeyData.metadata)}
                teamPolicy={readLensPolicy(lensTeam.data?.metadata)}
                teamPolicyStatus={inheritedLensStatus}
                canEdit={canEditLens}
                href={lensEntityHref({
                  kind: "key",
                  teamId: currentKeyData.team_id,
                  keyHash: currentKeyData.token_id || currentKeyData.token,
                })}
                onPolicyChange={updateLensPolicy}
              />
            </TabsContent>

            {/* Savings Panel. No keepMounted: this tab sweeps the daily rollup, and staying mounted
              would fire that request on every key page open for people who never look at it. */}
            <TabsContent value="savings">
              <KeySavingsTab
                accessToken={accessToken}
                keyToken={currentKeyData.token}
                userId={userID}
                userRole={userRole}
                activity={activityDateRange}
              />
            </TabsContent>

            {hasProxyWideSpendView(userRole) && (
              <TabsContent value="auto-router-usage">
                <KeyAutoRouterUsageTab
                  accessToken={accessToken}
                  keyToken={currentKeyData.token}
                  activity={activityDateRange}
                />
              </TabsContent>
            )}

            {/* Settings Panel */}
            <TabsContent value="settings" keepMounted>
              <Card className="block p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">{t("keyDetail.keyInfoView.settings.title")}</h3>
                  {!isEditing && canModifyKey && (
                    <Button variant="outline" onClick={() => setIsEditing(true)}>
                      {t("keyDetail.keyInfoView.settings.edit")}
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
                    editingMemberKeyAsTeamAdmin={editingMemberKeyAsTeamAdmin}
                  />
                ) : (
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.keyId")}</p>
                      <p className="text-sm font-mono">{currentKeyData.token_id || currentKeyData.token}</p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.keyAlias")}</p>
                      <p className="text-sm">
                        {currentKeyData.key_alias || t("keyDetail.keyInfoView.settings.notSet")}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.secretKey")}</p>
                      <p className="text-sm font-mono">{currentKeyData.key_name}</p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.teamId")}</p>
                      <p className="text-sm">
                        {currentKeyData.team_id ? (
                          <EntityLink href={teamDetailHref(currentKeyData.team_id)} className="font-normal">
                            {currentKeyData.team_id}
                          </EntityLink>
                        ) : (
                          t("keyDetail.keyInfoView.settings.notSet")
                        )}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.project")}</p>
                      <p className="text-sm">
                        {currentKeyData.project_id
                          ? (() => {
                              const project = projects?.find((p) => p.project_id === currentKeyData.project_id);
                              return project?.project_alias
                                ? `${project.project_alias} (${currentKeyData.project_id})`
                                : currentKeyData.project_id;
                            })()
                          : t("keyDetail.keyInfoView.settings.notSet")}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.organization")}</p>
                      <p className="text-sm">
                        {keyOrganizationId(currentKeyData) || t("keyDetail.keyInfoView.settings.notSet")}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.created")}</p>
                      <p className="text-sm">{formatTimestamp(currentKeyData.created_at)}</p>
                    </div>

                    {lastRegeneratedAt && (
                      <div>
                        <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.lastRegenerated")}</p>
                        <div className="flex items-center gap-2">
                          <p className="text-sm">{formatTimestamp(lastRegeneratedAt)}</p>
                          <Badge variant="secondary">{t("keyDetail.keyInfoView.settings.recent")}</Badge>
                        </div>
                      </div>
                    )}

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.expires")}</p>
                      <p className="text-sm">
                        {currentKeyData.expires
                          ? formatTimestamp(currentKeyData.expires)
                          : t("keyDetail.keyInfoView.settings.never")}
                      </p>
                    </div>

                    {Boolean(currentKeyData.metadata?.enable_prompt_caching) && (
                      <div>
                        <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.promptCaching")}</p>
                        <p className="text-sm">{t("keyDetail.keyInfoView.settings.promptCachingEnabled")}</p>
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
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.spend")}</p>
                      <p className="text-sm">
                        ${formatNumberWithCommas(currentKeyData.spend, 4)}
                        {t("keyDetail.keyInfoView.settings.usd")}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.lifetimeSpend")}</p>
                      <p className="text-sm">
                        ${formatNumberWithCommas(currentKeyData.total_spend ?? 0, 4)}
                        {t("keyDetail.keyInfoView.settings.usd")}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.budget")}</p>
                      <p className="text-sm">{budgetDisplay}</p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.budgetReset")}</p>
                      <p data-testid="budget-reset-value" className="text-sm">
                        {budgetResetTime
                          ? `${currentKeyData.budget_duration ? t("keyDetail.keyInfoView.settings.budgetResetEvery", { duration: currentKeyData.budget_duration }) : ""}${budgetResetTime}`
                          : t("keyDetail.keyInfoView.settings.never")}
                      </p>
                    </div>

                    {currentKeyData.budget_fallbacks && Object.keys(currentKeyData.budget_fallbacks).length > 0 && (
                      <div>
                        <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.budgetFallbacks")}</p>
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
                        <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.routerSettings")}</p>
                        <div className="mt-1">
                          <RouterSettingsSummary routerSettings={currentKeyData.router_settings} />
                        </div>
                      </div>
                    )}

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.tags")}</p>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {Array.isArray(currentKeyData.metadata?.tags) && currentKeyData.metadata.tags.length > 0
                          ? currentKeyData.metadata.tags.map((tag, index) => (
                              <span key={index} className="px-2 mr-2 py-1 bg-info/15 rounded-sm text-xs">
                                {tag}
                              </span>
                            ))
                          : t("keyDetail.keyInfoView.settings.noTags")}
                      </div>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.prompts")}</p>
                      <p className="text-sm">
                        {Array.isArray(currentKeyData.metadata?.prompts) && currentKeyData.metadata.prompts.length > 0
                          ? currentKeyData.metadata.prompts.map((prompt, index) => (
                              <span key={index} className="px-2 mr-2 py-1 bg-info/15 rounded-sm text-xs">
                                {prompt}
                              </span>
                            ))
                          : t("keyDetail.keyInfoView.settings.noPrompts")}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.allowedRoutes")}</p>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {Array.isArray(currentKeyData.allowed_routes) && currentKeyData.allowed_routes.length > 0 ? (
                          currentKeyData.allowed_routes.map((route, index) => (
                            <span key={index} className="px-2 py-1 bg-info/15 rounded-sm text-xs">
                              {route}
                            </span>
                          ))
                        ) : (
                          <Badge variant="secondary">{t("keyDetail.keyInfoView.settings.allRoutesAllowed")}</Badge>
                        )}
                      </div>
                    </div>

                    <div>
                      <p className="text-sm font-medium">
                        {t("keyDetail.keyInfoView.settings.allowedPassThroughRoutes")}
                      </p>
                      <p className="text-sm">
                        {Array.isArray(currentKeyData.metadata?.allowed_passthrough_routes) &&
                        currentKeyData.metadata.allowed_passthrough_routes.length > 0
                          ? currentKeyData.metadata.allowed_passthrough_routes.map((route, index) => (
                              <span key={index} className="px-2 mr-2 py-1 bg-info/15 rounded-sm text-xs">
                                {route}
                              </span>
                            ))
                          : t("keyDetail.keyInfoView.settings.noPassThroughRoutes")}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">
                        {t("keyDetail.keyInfoView.settings.disableGlobalGuardrails")}
                      </p>
                      <p className="text-sm">
                        {currentKeyData.metadata?.disable_global_guardrails === true ? (
                          <Badge variant="destructive">
                            {t("keyDetail.keyInfoView.settings.globalGuardrailsBypassed")}
                          </Badge>
                        ) : (
                          <Badge variant="secondary">
                            {t("keyDetail.keyInfoView.settings.globalGuardrailsActive")}
                          </Badge>
                        )}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.models")}</p>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {currentKeyData.models && currentKeyData.models.length > 0 ? (
                          currentKeyData.models.map((model, index) => (
                            <BadgeLink
                              key={index}
                              href={modelOrAccessGroupHref(model, accessGroupNames)}
                              className="min-w-0 break-words"
                            >
                              {model}
                            </BadgeLink>
                          ))
                        ) : (
                          <p className="text-sm">{t("keyDetail.keyInfoView.settings.noModels")}</p>
                        )}
                      </div>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.rateLimits")}</p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.tpm")}
                        {currentKeyData.tpm_limit !== null
                          ? currentKeyData.tpm_limit
                          : t("keyDetail.keyInfoView.settings.unlimited")}
                      </p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.rpm")}
                        {currentKeyData.rpm_limit !== null
                          ? currentKeyData.rpm_limit
                          : t("keyDetail.keyInfoView.settings.unlimited")}
                      </p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.tpd")}
                        {currentKeyData.tpd_limit ?? t("keyDetail.keyInfoView.settings.unlimited")}
                      </p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.maxParallelRequests")}{" "}
                        {currentKeyData.max_parallel_requests !== null
                          ? currentKeyData.max_parallel_requests
                          : t("keyDetail.keyInfoView.settings.unlimited")}
                      </p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.modelTpmLimits")}{" "}
                        {currentKeyData.metadata?.model_tpm_limit
                          ? JSON.stringify(currentKeyData.metadata.model_tpm_limit)
                          : t("keyDetail.keyInfoView.settings.unlimited")}
                      </p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.modelRpmLimits")}{" "}
                        {currentKeyData.metadata?.model_rpm_limit
                          ? JSON.stringify(currentKeyData.metadata.model_rpm_limit)
                          : t("keyDetail.keyInfoView.settings.unlimited")}
                      </p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.tagRpmLimits")}{" "}
                        {currentKeyData.metadata?.tag_rpm_limit &&
                        Object.keys(currentKeyData.metadata.tag_rpm_limit).length > 0
                          ? JSON.stringify(currentKeyData.metadata.tag_rpm_limit)
                          : t("keyDetail.keyInfoView.settings.unlimited")}
                      </p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.estimatedOutputTokens")}{" "}
                        {currentKeyData.metadata?.default_estimated_output_tokens != null
                          ? String(currentKeyData.metadata.default_estimated_output_tokens)
                          : t("keyDetail.keyInfoView.settings.default")}
                      </p>
                      <p className="text-sm">
                        {t("keyDetail.keyInfoView.settings.estimatedOutputTokensPerModel")}{" "}
                        {currentKeyData.metadata?.default_estimated_output_tokens_per_model
                          ? JSON.stringify(currentKeyData.metadata.default_estimated_output_tokens_per_model)
                          : t("keyDetail.keyInfoView.settings.default")}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm font-medium">{t("keyDetail.keyInfoView.settings.metadata")}</p>
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
    </div>
  );
}
