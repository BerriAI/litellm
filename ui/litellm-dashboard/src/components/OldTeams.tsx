import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import AvailableTeamsPanel from "@/components/team/available_teams";
import TeamInfoView from "@/components/team/TeamInfo";
import TeamSSOSettings from "@/components/TeamSSOSettings";
import { isProxyAdminRole } from "@/utils/roles";
import { useTranslation } from "react-i18next";
import { InfoCircleOutlined, PlusOutlined, TeamOutlined, ReloadOutlined } from "@ant-design/icons";
import { Accordion, AccordionBody, AccordionHeader, TextInput } from "@tremor/react";
import {
  Button,
  Card,
  Flex,
  Form,
  Input,
  Layout,
  Modal,
  Pagination,
  Progress,
  Select,
  Space,
  Switch,
  Table,
  Tabs,
  Tag,
  theme,
  Tooltip,
  Typography,
  message,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import type { SorterResult } from "antd/es/table/interface";
import { KeyIcon, LayersIcon, SearchIcon, UsersIcon } from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { AntDLoadingSpinner } from "@/components/ui/AntDLoadingSpinner";
import OrganizationDropdown from "./common_components/OrganizationDropdown";
import TableIconActionButton from "./common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { teamListCall as v2TeamListCall, type TeamsResponse } from "@/app/(dashboard)/hooks/teams/useTeams";
import AccessGroupSelector from "./common_components/AccessGroupSelector";
import PassThroughRoutesSelector from "./common_components/PassThroughRoutesSelector";
import AgentSelector from "./agent_management/AgentSelector";
import ModelAliasManager from "./common_components/ModelAliasManager";
import PremiumLoggingSettings from "./common_components/PremiumLoggingSettings";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "./common_components/RouterSettingsAccordion";
import {
  fetchAvailableModelsForTeamOrKey,
  unfurlWildcardModelsInList,
} from "./key_team_helpers/fetch_available_models_team_key";
import type { KeyResponse, Team } from "./key_team_helpers/key_list";
import MCPServerSelector from "./mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "./mcp_server_management/MCPToolPermissions";
import NotificationsManager from "./molecules/notifications_manager";
import { Organization, fetchMCPAccessGroups, getGuardrailsList, getPoliciesList, teamDeleteCall } from "./networking";
import NumericalInput from "./shared/numerical_input";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import SearchToolSelector from "./SearchTools/SearchToolSelector";

interface TeamProps {
  teams: Team[] | null;
  searchParams: any;
  accessToken: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Team[] | null>>;
  userID: string | null;
  userRole: string | null;
  organizations: Organization[] | null;
  premiumUser?: boolean;
}

interface FilterState {
  search: string;
  organization_id: string;
  sort_by: string;
  sort_order: "asc" | "desc";
}

interface EditTeamModalProps {
  visible: boolean;
  onCancel: () => void;
  team: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
}

import { updateExistingKeys } from "@/utils/dataUtils";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
import { Member, teamCreateCall } from "./networking";
import { ModelSelect } from "./ModelSelect/ModelSelect";

interface TeamInfo {
  members_with_roles: Member[];
}

interface PerTeamInfo {
  keys: KeyResponse[];
  keys_count: number;
  team_info: TeamInfo;
}

const getOrganizationModels = (organization: Organization | null, userModels: string[]) => {
  let tempModelsToPick = [];

  if (organization) {
    if (organization.models.length > 0) {
      console.log(`organization.models: ${organization.models}`);
      tempModelsToPick = organization.models;
    } else {
      // show all available models if the team has no models set
      tempModelsToPick = userModels;
    }
  } else {
    // no team set, show all available models
    tempModelsToPick = userModels;
  }

  return unfurlWildcardModelsInList(tempModelsToPick, userModels);
};

const canCreateOrManageTeams = (
  userRole: string | null,
  userID: string | null,
  organizations: Organization[] | null,
): boolean => {
  // Admin role always has permission
  if (userRole === "Admin") {
    return true;
  }

  // Check if user is an org_admin in any organization
  if (organizations && userID) {
    return organizations.some((org) =>
      org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
    );
  }

  return false;
};

const getAdminOrganizations = (
  userRole: string | null,
  userID: string | null,
  organizations: Organization[] | null,
): Organization[] => {
  // Global Admin can see all organizations
  if (userRole === "Admin") {
    return organizations || [];
  }

  // Org Admin can only see organizations they're an admin for
  if (organizations && userID) {
    return organizations.filter((org) =>
      org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
    );
  }

  return [];
};

const getOrganizationAlias = (
  organizationId: string | null | undefined,
  organizations: Organization[] | null | undefined,
): string => {
  if (!organizationId || !organizations) {
    return organizationId || "N/A";
  }

  const organization = organizations.find((org) => org.organization_id === organizationId);
  return organization?.organization_alias || organizationId;
};

// @deprecated
const Teams: React.FC<TeamProps> = ({
  teams,
  searchParams,
  accessToken,
  setTeams,
  userID,
  userRole,
  organizations,
  premiumUser = false,
}) => {
  console.log(`organizations: ${JSON.stringify(organizations)}`);
  const { t } = useTranslation();
  const { data: organizationsData } = useOrganizations();
  const [isLoading, setIsLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalTeams, setTotalTeams] = useState(0);
  const [currentOrg, setCurrentOrg] = useState<Organization | null>(null);
  const [currentOrgForCreateTeam, setCurrentOrgForCreateTeam] = useState<Organization | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    search: "",
    organization_id: "",
    sort_by: "created_at",
    sort_order: "desc",
  });
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  const fetchTeamsV2 = async (
    opts: {
      page?: number;
      size?: number;
      sortBy?: string;
      sortOrder?: string;
      organizationID?: string;
      search?: string;
    } = {},
  ) => {
    if (!accessToken) return;
    const page = opts.page ?? currentPage;
    const size = opts.size ?? pageSize;
    const sortBy = opts.sortBy ?? filters.sort_by;
    const sortOrder = opts.sortOrder ?? filters.sort_order;
    const organizationID = opts.organizationID ?? filters.organization_id;
    const search = opts.search ?? filters.search;

    setIsLoading(true);
    setFetchError(null);
    try {
      const response: TeamsResponse = await v2TeamListCall(accessToken, page, size, {
        organizationID: organizationID || null,
        search: search || null,
        userID: userRole !== "Admin" && userRole !== "Admin Viewer" ? userID : null,
        sortBy: sortBy || null,
        sortOrder: sortOrder || null,
      });
      setTeams(response.teams ?? []);
      setTotalTeams(response.total ?? 0);
    } catch (err: any) {
      setFetchError(err?.message || t("oldTeams.failedToFetchTeams"));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTeamsV2();
  }, [accessToken]);

  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();
  const [value, setValue] = useState("");
  const [editModalVisible, setEditModalVisible] = useState(false);

  const [selectedTeam, setSelectedTeam] = useState<null | any>(null);
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [editTeam, setEditTeam] = useState<boolean>(false);

  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [teamToDelete, setTeamToDelete] = useState<Team | null>(null);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [perTeamInfo, setPerTeamInfo] = useState<Record<string, PerTeamInfo>>({});
  const [isTeamDeleting, setIsTeamDeleting] = useState(false);
  // Add this state near the other useState declarations
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [policiesList, setPoliciesList] = useState<string[]>([]);
  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [mcpAccessGroups, setMcpAccessGroups] = useState<string[]>([]);
  const [mcpAccessGroupsLoaded, setMcpAccessGroupsLoaded] = useState(false);
  const [modelAliases, setModelAliases] = useState<{ [key: string]: string }>({});
  const [routerSettings, setRouterSettings] = useState<RouterSettingsAccordionValue | null>(null);
  const [routerSettingsKey, setRouterSettingsKey] = useState<number>(0);

  useEffect(() => {
    console.log(`currentOrgForCreateTeam: ${currentOrgForCreateTeam}`);
    const models = getOrganizationModels(currentOrgForCreateTeam, userModels);
    console.log(`models: ${models}`);
    setModelsToPick(models);
    form.setFieldValue("models", []);
  }, [currentOrgForCreateTeam, userModels]);

  // Handle organization preselection when modal opens
  useEffect(() => {
    if (isTeamModalVisible) {
      const adminOrgs = getAdminOrganizations(userRole, userID, organizations);

      // If there's exactly one organization the user is admin for, preselect it
      if (adminOrgs.length === 1) {
        const org = adminOrgs[0];
        form.setFieldValue("organization_id", org.organization_id);
        setCurrentOrgForCreateTeam(org);
      } else {
        // Reset the organization selection for multiple orgs
        form.setFieldValue("organization_id", currentOrg?.organization_id || null);
        setCurrentOrgForCreateTeam(currentOrg);
      }
    }
  }, [isTeamModalVisible, userRole, userID, organizations, currentOrg]);

  // Add this useEffect to fetch guardrails
  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        if (accessToken == null) {
          return;
        }

        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    const fetchPolicies = async () => {
      try {
        if (accessToken == null) {
          return;
        }

        const response = await getPoliciesList(accessToken);
        const policyNames = response.policies.map((p: { policy_name: string }) => p.policy_name);
        setPoliciesList(policyNames);
      } catch (error) {
        console.error("Failed to fetch policies:", error);
      }
    };

    fetchGuardrails();
    fetchPolicies();
  }, [accessToken]);

  const fetchMcpAccessGroups = async () => {
    try {
      if (accessToken == null) {
        return;
      }
      const groups = await fetchMCPAccessGroups(accessToken);
      setMcpAccessGroups(groups);
    } catch (error) {
      console.error("Failed to fetch MCP access groups:", error);
    }
  };

  useEffect(() => {
    fetchMcpAccessGroups();
  }, [accessToken]);

  useEffect(() => {
    const fetchTeamInfo = () => {
      if (!teams) return;

      const newPerTeamInfo = teams.reduce(
        (acc, team) => {
          acc[team.team_id] = {
            keys: team.keys || [],
            keys_count: team.keys_count ?? team.keys?.length ?? 0,
            team_info: {
              members_with_roles: team.members_with_roles || [],
            },
          };
          return acc;
        },
        {} as Record<string, PerTeamInfo>,
      );

      setPerTeamInfo(newPerTeamInfo);
    };

    fetchTeamInfo();
  }, [teams]);

  const handleOk = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setModelAliases({});
    setRouterSettings(null);
    setRouterSettingsKey((prev) => prev + 1);
  };

  const handleMemberOk = () => {
    setIsAddMemberModalVisible(false);
    setIsEditMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleCancel = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setModelAliases({});
    setRouterSettings(null);
    setRouterSettingsKey((prev) => prev + 1);
  };

  const handleMemberCancel = () => {
    setIsAddMemberModalVisible(false);
    setIsEditMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleDelete = async (team: Team) => {
    // Set the team to delete and open the confirmation modal
    setTeamToDelete(team);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (teamToDelete == null || teams == null || accessToken == null) {
      return;
    }

    try {
      setIsTeamDeleting(true);
      await teamDeleteCall(accessToken, teamToDelete.team_id);
      await fetchTeamsV2();
      NotificationsManager.success(t("oldTeams.teamDeletedSuccess"));
    } catch (error) {
      NotificationsManager.fromBackend(t("oldTeams.deleteTeamFailed", { error }));
    } finally {
      setIsTeamDeleting(false);
      setIsDeleteModalOpen(false);
      setTeamToDelete(null);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setTeamToDelete(null);
  };

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null || accessToken === null) {
          return;
        }
        const models = await fetchAvailableModelsForTeamOrKey(userID, userRole, accessToken);
        if (models) {
          setUserModels(models);
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole, teams]);

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      console.log(`formValues: ${JSON.stringify(formValues)}`);
      if (accessToken != null) {
        const newTeamAlias = formValues?.team_alias;
        const existingTeamAliases = teams?.map((t) => t.team_alias) ?? [];
        let organizationId = formValues?.organization_id || currentOrg?.organization_id;
        if (organizationId === "" || typeof organizationId !== "string") {
          formValues.organization_id = null;
        } else {
          formValues.organization_id = organizationId.trim();
        }

        // Remove guardrails from top level since it's now in metadata
        if (existingTeamAliases.includes(newTeamAlias)) {
          throw new Error(`Team alias ${newTeamAlias} already exists, please pick another alias`);
        }

        NotificationsManager.info(t("oldTeams.creatingTeam"));

        // Handle logging settings in metadata
        if (loggingSettings.length > 0) {
          let metadata = {};
          if (formValues.metadata) {
            try {
              metadata = JSON.parse(formValues.metadata);
            } catch (e) {
              console.warn("Invalid JSON in metadata field, starting with empty object");
            }
          }

          // Add logging settings to metadata
          metadata = {
            ...metadata,
            logging: loggingSettings.filter((config) => config.callback_name), // Only include configs with callback_name
          };

          formValues.metadata = JSON.stringify(metadata);
        }

        if (formValues.secret_manager_settings) {
          if (typeof formValues.secret_manager_settings === "string") {
            if (formValues.secret_manager_settings.trim() === "") {
              delete formValues.secret_manager_settings;
            } else {
              try {
                formValues.secret_manager_settings = JSON.parse(formValues.secret_manager_settings);
              } catch (e) {
                throw new Error("Failed to parse secret manager settings: " + e);
              }
            }
          }
        }

        const hasSearchTools =
          Array.isArray(formValues.object_permission_search_tools) &&
          formValues.object_permission_search_tools.length > 0;

        if (
          (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) ||
          (formValues.allowed_mcp_servers_and_groups &&
            (formValues.allowed_mcp_servers_and_groups.servers?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.accessGroups?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.toolPermissions))
        ) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          if (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) {
            formValues.object_permission.vector_stores = formValues.allowed_vector_store_ids;
            delete formValues.allowed_vector_store_ids;
          }
          if (formValues.allowed_mcp_servers_and_groups) {
            const { servers, accessGroups } = formValues.allowed_mcp_servers_and_groups;
            if (servers && servers.length > 0) {
              formValues.object_permission.mcp_servers = servers;
            }
            if (accessGroups && accessGroups.length > 0) {
              formValues.object_permission.mcp_access_groups = accessGroups;
            }
            delete formValues.allowed_mcp_servers_and_groups;
          }

          if (formValues.mcp_tool_permissions && Object.keys(formValues.mcp_tool_permissions).length > 0) {
            formValues.object_permission.mcp_tool_permissions = formValues.mcp_tool_permissions;
            delete formValues.mcp_tool_permissions;
          }
        }

        // Transform allowed_mcp_access_groups into object_permission
        if (formValues.allowed_mcp_access_groups && formValues.allowed_mcp_access_groups.length > 0) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          formValues.object_permission.mcp_access_groups = formValues.allowed_mcp_access_groups;
          delete formValues.allowed_mcp_access_groups;
        }

        // Handle agent permissions
        if (formValues.allowed_agents_and_groups) {
          const { agents, accessGroups } = formValues.allowed_agents_and_groups;
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          if (agents && agents.length > 0) {
            formValues.object_permission.agents = agents;
          }
          if (accessGroups && accessGroups.length > 0) {
            formValues.object_permission.agent_access_groups = accessGroups;
          }
          delete formValues.allowed_agents_and_groups;
        }

        if (hasSearchTools) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          formValues.object_permission.search_tools = formValues.object_permission_search_tools;
          delete formValues.object_permission_search_tools;
        }

        // Add model_aliases if any are defined
        if (Object.keys(modelAliases).length > 0) {
          formValues.model_aliases = modelAliases;
        }

        // Add router_settings if any are defined
        if (routerSettings?.router_settings) {
          // Only include router_settings if it has at least one non-null value
          const hasValues = Object.values(routerSettings.router_settings).some(
            (value) => value !== null && value !== undefined && value !== "",
          );
          if (hasValues) {
            formValues.router_settings = routerSettings.router_settings;
          }
        }

        await teamCreateCall(accessToken, formValues);
        NotificationsManager.success(t("oldTeams.teamCreatedSuccess"));
        await fetchTeamsV2({
          page: currentPage,
          size: pageSize,
        });
        form.resetFields();
        setLoggingSettings([]);
        setModelAliases({});
        setRouterSettings(null);
        setRouterSettingsKey((prev) => prev + 1);
        setIsTeamModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the team:", error);
      NotificationsManager.fromBackend(t("oldTeams.createTeamFailed", { error }));
    }
  };

  const is_team_admin = (team: any) => {
    if (team == null || team.members_with_roles == null) {
      return false;
    }
    for (let i = 0; i < team.members_with_roles.length; i++) {
      let member = team.members_with_roles[i];
      if (member.user_id == userID && member.role == "admin") {
        return true;
      }
    }
    return false;
  };

  const handleSearchChange = (value: string) => {
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    setIsSearching(true);
    searchDebounceRef.current = setTimeout(async () => {
      try {
        setFilters((prev) => ({ ...prev, search: value }));
        setCurrentPage(1);
        await fetchTeamsV2({ page: 1, search: value });
      } finally {
        setIsSearching(false);
      }
    }, 300);
  };

  const handleFilterChange = async (key: keyof FilterState, value: string) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    setCurrentPage(1);
    if (!accessToken) return;
    try {
      const response: TeamsResponse = await v2TeamListCall(accessToken, 1, pageSize, {
        organizationID: newFilters.organization_id || null,
        search: newFilters.search || null,
        userID: userRole !== "Admin" && userRole !== "Admin Viewer" ? userID : null,
        sortBy: newFilters.sort_by || null,
        sortOrder: newFilters.sort_order || null,
      });
      setTeams(response.teams ?? []);
      setTotalTeams(response.total ?? 0);
    } catch (error) {
      console.error("Error fetching teams:", error);
    }
  };

  const handleFilterReset = () => {
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    setIsSearching(false);
    const resetFilters: FilterState = {
      search: "",
      organization_id: "",
      sort_by: "created_at",
      sort_order: "desc",
    };
    setFilters(resetFilters);
    setCurrentPage(1);
    fetchTeamsV2({ page: 1, organizationID: "", search: "", sortBy: "created_at", sortOrder: "desc" });
  };

  const { token } = theme.useToken();
  const { Title, Text } = Typography;
  const { Content } = Layout;

  const handleRetry = () => {
    fetchTeamsV2();
  };

  const handleTableSort = (
    _pagination: unknown,
    _filters: unknown,
    sorter: SorterResult<Team> | SorterResult<Team>[],
  ) => {
    const s = Array.isArray(sorter) ? sorter[0] : sorter;
    const sortBy = s.order ? (s.columnKey as string) : "created_at";
    const sortOrder = s.order === "ascend" ? "asc" : s.order === "descend" ? "desc" : "desc";
    setFilters((prev) => ({ ...prev, sort_by: sortBy, sort_order: sortOrder }));
    fetchTeamsV2({ sortBy, sortOrder });
  };

  const teamColumns: ColumnsType<Team> = useMemo(
    () => [
      {
        title: t("oldTeams.columns.teamId"),
        dataIndex: "team_id",
        key: "team_id",
        width: 170,
        ellipsis: true,
        render: (id: string, record: Team) => (
          <Tooltip title={id}>
            <Text
              ellipsis
              className="text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs cursor-pointer"
              style={{ fontSize: 14, padding: "1px 8px" }}
              onClick={() => setSelectedTeamId(record.team_id)}
              data-testid="team-id-cell"
            >
              {id}
            </Text>
          </Tooltip>
        ),
      },
      {
        title: t("oldTeams.columns.teamAlias"),
        dataIndex: "team_alias",
        key: "team_alias",
        ellipsis: true,
        sorter: true,
        render: (alias: string | undefined) => (
          <Text style={{ fontSize: 14 }}>
            {alias || (
              <Text type="secondary" italic>
                —
              </Text>
            )}
          </Text>
        ),
      },
      {
        title: t("oldTeams.columns.organization"),
        key: "organization",
        width: 160,
        ellipsis: true,
        render: (_: unknown, record: Team) => {
          const orgAlias = getOrganizationAlias(record.organization_id, organizationsData || organizations);
          return record.organization_id ? (
            <Text ellipsis style={{ fontSize: 14 }}>
              {orgAlias}
            </Text>
          ) : (
            <Text type="secondary">—</Text>
          );
        },
      },
      {
        title: t("oldTeams.columns.resources"),
        key: "resources",
        width: 240,
        render: (_: unknown, record: Team) => {
          const memberCount = perTeamInfo?.[record.team_id]?.team_info?.members_with_roles?.length ?? 0;
          const modelCount = record.models?.length ?? 0;
          const keyCount = perTeamInfo?.[record.team_id]?.keys_count ?? 0;
          return (
            <Flex gap={12} align="center">
              <Tooltip title={t("oldTeams.resources.members", { count: memberCount })}>
                <Tag color="purple" style={{ fontSize: 14, padding: "2px 8px", margin: 0 }}>
                  <Flex align="center" gap={6}>
                    <UsersIcon size={14} />
                    {memberCount}
                  </Flex>
                </Tag>
              </Tooltip>
              <Tooltip title={t("oldTeams.resources.models", { count: modelCount })}>
                <Tag color="blue" style={{ fontSize: 14, padding: "2px 8px", margin: 0 }}>
                  <Flex align="center" gap={6}>
                    <LayersIcon size={14} />
                    {modelCount}
                  </Flex>
                </Tag>
              </Tooltip>
              <Tooltip title={t("oldTeams.resources.keys", { count: keyCount })}>
                <Tag color="cyan" style={{ fontSize: 14, padding: "2px 8px", margin: 0 }}>
                  <Flex align="center" gap={6}>
                    <KeyIcon size={14} />
                    {keyCount}
                  </Flex>
                </Tag>
              </Tooltip>
            </Flex>
          );
        },
      },
      {
        title: t("oldTeams.columns.spendBudget"),
        key: "spend",
        width: 200,
        sorter: true,
        render: (_: unknown, record: Team) => {
          const spendVal = record.spend ?? 0;
          const budgetVal = record.max_budget;
          const spendStr = `$${spendVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
          const budgetStr =
            budgetVal != null
              ? `$${budgetVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
              : t("oldTeams.unlimited");
          const percent = budgetVal != null && budgetVal > 0 ? Math.min((spendVal / budgetVal) * 100, 100) : null;
          return (
            <Flex vertical gap={2}>
              <Text style={{ fontSize: 13 }}>
                {spendStr}
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {" / "}
                  {budgetStr}
                </Text>
              </Text>
              {percent != null && (
                <Progress
                  percent={percent}
                  size="small"
                  showInfo={false}
                  strokeColor={percent >= 90 ? "#ff4d4f" : percent >= 70 ? "#faad14" : "#1677ff"}
                  style={{ marginBottom: 0 }}
                />
              )}
            </Flex>
          );
        },
      },
      {
        title: t("oldTeams.columns.created"),
        dataIndex: "created_at",
        key: "created_at",
        width: 130,
        ellipsis: true,
        sorter: true,
        render: (date: string | undefined) => (
          <Text type="secondary" style={{ fontSize: 13 }}>
            {date
              ? new Date(date).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })
              : "—"}
          </Text>
        ),
      },
      {
        title: t("common.actions"),
        key: "actions",
        width: 120,
        align: "right" as const,
        render: (_: unknown, record: Team) => (
          <Space size={4}>
            <TableIconActionButton
              variant="Copy"
              tooltipText={t("oldTeams.actions.copyTeamId")}
              onClick={() => {
                navigator.clipboard
                  .writeText(record.team_id)
                  .then(() => message.success(t("oldTeams.actions.teamIdCopied")))
                  .catch(() => message.error(t("oldTeams.actions.failedToCopy")));
              }}
            />
            {userRole === "Admin" && (
              <>
                <TableIconActionButton
                  variant="Edit"
                  tooltipText={t("oldTeams.actions.editTeam")}
                  dataTestId="edit-team-button"
                  onClick={() => {
                    setSelectedTeamId(record.team_id);
                    setEditTeam(true);
                  }}
                />
                <TableIconActionButton
                  variant="Delete"
                  tooltipText={t("oldTeams.actions.deleteTeam")}
                  dataTestId="delete-team-button"
                  onClick={() => handleDelete(record)}
                />
              </>
            )}
          </Space>
        ),
      },
    ],
    [userRole, perTeamInfo, organizationsData, organizations, t],
  );

  const displayTeams = useMemo(() => teams ?? [], [teams]);

  const renderTeamsContent = () => {
    if (isLoading) {
      return (
        <Flex justify="center" align="center" style={{ padding: "80px 0" }}>
          <AntDLoadingSpinner fontSize={48} />
        </Flex>
      );
    }

    if (fetchError) {
      return (
        <Flex vertical align="center" gap={16} style={{ padding: "64px 0" }}>
          <Text type="danger" style={{ fontSize: 15 }}>
            {t("oldTeams.failedToLoadTeams")}
          </Text>
          <Text type="secondary" style={{ fontSize: 13 }}>
            {fetchError}
          </Text>
          <Button icon={<ReloadOutlined />} onClick={handleRetry}>
            {t("common.retry")}
          </Button>
        </Flex>
      );
    }

    return (
      <Table<Team>
        columns={teamColumns}
        dataSource={displayTeams}
        rowKey="team_id"
        pagination={false}
        onChange={handleTableSort}
        locale={{
          emptyText: (
            <div style={{ padding: "64px 0", textAlign: "center" }}>
              <TeamOutlined style={{ fontSize: 40, color: "#d9d9d9", marginBottom: 12 }} />
              <div>
                <Text style={{ fontSize: 15, color: "#595959" }}>{t("oldTeams.noTeamsYet")}</Text>
              </div>
              <div style={{ marginTop: 4 }}>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  {t("oldTeams.createFirstTeamHint")}
                </Text>
              </div>
              {canCreateOrManageTeams(userRole, userID, organizations) && (
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setIsTeamModalVisible(true)}
                  style={{ marginTop: 16 }}
                  data-testid="create-team-button"
                >
                  {t("oldTeams.createTeam")}
                </Button>
              )}
            </div>
          ),
        }}
        scroll={{ x: 1000 }}
        size="middle"
      />
    );
  };

  const tabItems = [
    {
      key: "your-teams",
      label: t("oldTeams.tabs.yourTeams"),
      children: (
        <>
          <Card styles={{ body: { padding: 0 } }}>
            <Flex justify="space-between" align="center" style={{ padding: "12px 16px" }}>
              <Flex gap={12} align="center">
                <Input
                  prefix={<SearchIcon size={16} />}
                  suffix={isSearching ? <AntDLoadingSpinner size="small" /> : null}
                  placeholder={t("oldTeams.searchPlaceholder")}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  allowClear
                  style={{ maxWidth: 400 }}
                />
                <OrganizationDropdown
                  organizations={organizations}
                  value={filters.organization_id || undefined}
                  onChange={(value: string) => handleFilterChange("organization_id", value || "")}
                  loading={isLoading}
                />
              </Flex>
              <Pagination
                current={currentPage}
                total={totalTeams}
                pageSize={pageSize}
                onChange={(page, size) => {
                  setCurrentPage(page);
                  setPageSize(size);
                  fetchTeamsV2({ page, size });
                }}
                size="small"
                showTotal={(total) => t("oldTeams.totalTeams", { count: total })}
                showSizeChanger
                pageSizeOptions={["10", "20", "50"]}
              />
            </Flex>

            {renderTeamsContent()}
          </Card>

          <DeleteResourceModal
            isOpen={isDeleteModalOpen}
            title={t("oldTeams.deleteModal.title")}
            alertMessage={(() => {
              const deleteKeyCount = teamToDelete?.keys_count ?? teamToDelete?.keys?.length ?? 0;
              return deleteKeyCount === 0
                ? undefined
                : t("oldTeams.deleteModal.alertMessage", { count: deleteKeyCount });
            })()}
            message={t("oldTeams.deleteModal.message")}
            resourceInformationTitle={t("oldTeams.deleteModal.resourceInformationTitle")}
            resourceInformation={[
              { label: t("oldTeams.deleteModal.labelTeamId"), value: teamToDelete?.team_id, code: true },
              { label: t("oldTeams.deleteModal.labelTeamName"), value: teamToDelete?.team_alias },
              {
                label: t("oldTeams.deleteModal.labelKeys"),
                value: teamToDelete?.keys_count ?? teamToDelete?.keys?.length ?? 0,
              },
              { label: t("oldTeams.deleteModal.labelMembers"), value: teamToDelete?.members_with_roles?.length },
            ]}
            requiredConfirmation={teamToDelete?.team_alias}
            onCancel={cancelDelete}
            onOk={confirmDelete}
            confirmLoading={isTeamDeleting}
          />
        </>
      ),
    },
    {
      key: "available-teams",
      label: t("oldTeams.tabs.availableTeams"),
      children: <AvailableTeamsPanel accessToken={accessToken} userID={userID} />,
    },
    ...(isProxyAdminRole(userRole || "")
      ? [
          {
            key: "default-settings",
            label: t("oldTeams.tabs.defaultTeamSettings"),
            children: <TeamSSOSettings accessToken={accessToken} userID={userID || ""} userRole={userRole || ""} />,
          },
        ]
      : []),
  ];

  return (
    <Content style={{ padding: token.paddingLG, paddingInline: token.paddingLG * 2 }}>
      {selectedTeamId ? (
        <TeamInfoView
          teamId={selectedTeamId}
          onUpdate={(data) => {
            setTeams((teams) => {
              if (teams == null) {
                return teams;
              }
              return teams.map((team) => {
                if (data.team_id === team.team_id) {
                  return updateExistingKeys(team, data);
                }
                return team;
              });
            });
            fetchTeamsV2();
          }}
          onClose={() => {
            setSelectedTeamId(null);
            setEditTeam(false);
          }}
          accessToken={accessToken}
          is_team_admin={is_team_admin(teams?.find((team) => team.team_id === selectedTeamId))}
          is_proxy_admin={userRole == "Admin"}
          userModels={userModels}
          editTeam={editTeam}
          premiumUser={premiumUser}
        />
      ) : (
        <>
          <Flex justify="space-between" align="center" style={{ marginBottom: 16 }}>
            <Space direction="vertical" size={0}>
              <Title level={2} style={{ margin: 0 }}>
                <TeamOutlined style={{ marginRight: 8 }} />
                {t("oldTeams.pageTitle")}
              </Title>
              <Text type="secondary">{t("oldTeams.pageSubtitle")}</Text>
            </Space>
            {canCreateOrManageTeams(userRole, userID, organizations) && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setIsTeamModalVisible(true)}
                data-testid="create-team-button"
              >
                {t("oldTeams.createTeam")}
              </Button>
            )}
          </Flex>

          <Tabs items={tabItems} />
        </>
      )}

      {canCreateOrManageTeams(userRole, userID, organizations) && (
        <Modal
          title={t("oldTeams.createTeam")}
          open={isTeamModalVisible}
          width={1000}
          footer={null}
          onOk={handleOk}
          onCancel={handleCancel}
        >
          <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
            <>
              <Form.Item
                label={t("oldTeams.form.teamName")}
                name="team_alias"
                rules={[
                  {
                    required: true,
                    message: t("oldTeams.form.teamNameRequired"),
                  },
                ]}
              >
                <TextInput placeholder="" data-testid="team-name-input" />
              </Form.Item>
              {(() => {
                const adminOrgs = getAdminOrganizations(userRole, userID, organizations);
                const isOrgAdmin = userRole !== "Admin";
                const isSingleOrg = adminOrgs.length === 1;
                const hasNoOrgs = adminOrgs.length === 0;

                return (
                  <>
                    <Form.Item
                      label={
                        <span>
                          {t("oldTeams.form.organization")}{" "}
                          <Tooltip
                            title={
                              <span>
                                {t("oldTeams.form.organizationTooltipPrefix")}{" "}
                                <a
                                  href="https://docs.litellm.ai/docs/proxy/user_management_heirarchy"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  style={{
                                    color: "#1890ff",
                                    textDecoration: "underline",
                                  }}
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  {t("oldTeams.form.userManagementHierarchy")}
                                </a>
                              </span>
                            }
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                      name="organization_id"
                      initialValue={currentOrg ? currentOrg.organization_id : null}
                      className="mt-8"
                      rules={
                        isOrgAdmin
                          ? [
                              {
                                required: true,
                                message: t("oldTeams.form.organizationRequired"),
                              },
                            ]
                          : []
                      }
                      help={isSingleOrg ? t("oldTeams.form.singleOrgHelp") : isOrgAdmin ? t("common.required") : ""}
                    >
                      <Select
                        showSearch
                        allowClear={!isOrgAdmin}
                        disabled={isSingleOrg}
                        placeholder={
                          hasNoOrgs
                            ? t("oldTeams.form.noOrganizationsAvailable")
                            : t("oldTeams.form.searchOrSelectOrganization")
                        }
                        onChange={(value) => {
                          form.setFieldValue("organization_id", value);
                          setCurrentOrgForCreateTeam(adminOrgs?.find((org) => org.organization_id === value) || null);
                        }}
                        filterOption={(input, option) => {
                          if (!option) return false;
                          const optionValue = option.children?.toString() || "";
                          return optionValue.toLowerCase().includes(input.toLowerCase());
                        }}
                        optionFilterProp="children"
                      >
                        {adminOrgs?.map((org) => (
                          <Select.Option key={org.organization_id} value={org.organization_id}>
                            <span className="font-medium">{org.organization_alias}</span>{" "}
                            <span className="text-gray-500">({org.organization_id})</span>
                          </Select.Option>
                        ))}
                      </Select>
                    </Form.Item>

                    {/* Show message when org admin needs to select organization */}
                    {isOrgAdmin && !isSingleOrg && adminOrgs.length > 1 && (
                      <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-md">
                        <Text style={{ color: "#1e40af", fontSize: 14 }}>{t("oldTeams.form.selectOrgAdminHint")}</Text>
                      </div>
                    )}
                  </>
                );
              })()}
              <Form.Item
                label={
                  <span>
                    {t("oldTeams.form.models")}{" "}
                    <Tooltip title={t("oldTeams.form.modelsTooltip")}>
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                rules={[
                  {
                    required: true,
                    message: t("oldTeams.form.modelsRequired"),
                  },
                ]}
                name="models"
              >
                <ModelSelect
                  value={form.getFieldValue("models") || []}
                  onChange={(values) => form.setFieldValue("models", values)}
                  organizationID={form.getFieldValue("organization_id")}
                  options={{
                    includeSpecialOptions: true,
                    showAllProxyModelsOverride: !form.getFieldValue("organization_id"),
                  }}
                  context="team"
                  dataTestId="create-team-models-select"
                />
              </Form.Item>

              <Form.Item label={t("oldTeams.form.maxBudget")} name="max_budget">
                <NumericalInput step={0.01} precision={2} width={200} />
              </Form.Item>
              <Form.Item className="mt-8" label={t("oldTeams.form.resetBudget")} name="budget_duration">
                <Select defaultValue={null} placeholder="n/a">
                  <Select.Option value="24h">{t("oldTeams.form.budgetDaily")}</Select.Option>
                  <Select.Option value="7d">{t("oldTeams.form.budgetWeekly")}</Select.Option>
                  <Select.Option value="30d">{t("oldTeams.form.budgetMonthly")}</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item label={t("oldTeams.form.tpmLimit")} name="tpm_limit">
                <NumericalInput step={1} width={400} />
              </Form.Item>
              <Form.Item label={t("oldTeams.form.rpmLimit")} name="rpm_limit">
                <NumericalInput step={1} width={400} />
              </Form.Item>

              <Accordion
                className="mt-20 mb-8"
                onClick={() => {
                  if (!mcpAccessGroupsLoaded) {
                    fetchMcpAccessGroups();
                    setMcpAccessGroupsLoaded(true);
                  }
                }}
              >
                <AccordionHeader>
                  <b>{t("oldTeams.accordions.additionalSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item label={t("oldTeams.form.teamId")} name="team_id" help={t("oldTeams.form.teamIdHelp")}>
                    <TextInput
                      onChange={(e) => {
                        e.target.value = e.target.value.trim();
                      }}
                    />
                  </Form.Item>
                  <Form.Item
                    label={t("oldTeams.form.teamMemberBudget")}
                    name="team_member_budget"
                    normalize={(value) => (value ? Number(value) : undefined)}
                    tooltip={t("oldTeams.form.teamMemberBudgetTooltip")}
                  >
                    <NumericalInput step={0.01} precision={2} width={200} />
                  </Form.Item>
                  <Form.Item
                    label={t("oldTeams.form.teamMemberKeyDuration")}
                    name="team_member_key_duration"
                    tooltip={t("oldTeams.form.teamMemberKeyDurationTooltip")}
                  >
                    <TextInput placeholder={t("oldTeams.form.teamMemberKeyDurationPlaceholder")} />
                  </Form.Item>
                  <Form.Item
                    label={t("oldTeams.form.teamMemberRpmLimit")}
                    name="team_member_rpm_limit"
                    tooltip={t("oldTeams.form.teamMemberRpmLimitTooltip")}
                  >
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <Form.Item
                    label={t("oldTeams.form.teamMemberTpmLimit")}
                    name="team_member_tpm_limit"
                    tooltip={t("oldTeams.form.teamMemberTpmLimitTooltip")}
                  >
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <Form.Item label={t("oldTeams.form.metadata")} name="metadata" help={t("oldTeams.form.metadataHelp")}>
                    <Input.TextArea rows={4} />
                  </Form.Item>
                  <Form.Item
                    label={t("oldTeams.form.secretManagerSettings")}
                    name="secret_manager_settings"
                    help={
                      premiumUser
                        ? t("oldTeams.form.secretManagerSettingsHelp")
                        : t("oldTeams.form.secretManagerSettingsPremiumHelp")
                    }
                    rules={[
                      {
                        validator: async (_, value) => {
                          if (!value) {
                            return Promise.resolve();
                          }
                          try {
                            JSON.parse(value);
                            return Promise.resolve();
                          } catch (error) {
                            return Promise.reject(new Error(t("oldTeams.form.invalidJson")));
                          }
                        },
                      },
                    ]}
                  >
                    <Input.TextArea
                      rows={4}
                      placeholder='{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}'
                      disabled={!premiumUser}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("oldTeams.form.guardrails")}{" "}
                        <Tooltip title={t("oldTeams.form.guardrailsTooltip")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="guardrails"
                    className="mt-8"
                    help={t("oldTeams.form.guardrailsHelp")}
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      placeholder={t("oldTeams.form.guardrailsPlaceholder")}
                      options={guardrailsList.map((name) => ({
                        value: name,
                        label: name,
                      }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("oldTeams.form.disableGlobalGuardrails")}{" "}
                        <Tooltip title={t("oldTeams.form.disableGlobalGuardrailsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="disable_global_guardrails"
                    className="mt-4"
                    valuePropName="checked"
                    help={t("oldTeams.form.disableGlobalGuardrailsHelp")}
                  >
                    <Switch
                      disabled={!premiumUser}
                      checkedChildren={
                        premiumUser ? t("common.yes") : t("oldTeams.form.disableGlobalGuardrailsPremium")
                      }
                      unCheckedChildren={
                        premiumUser ? t("common.no") : t("oldTeams.form.disableGlobalGuardrailsPremium")
                      }
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("oldTeams.form.policies")}{" "}
                        <Tooltip title={t("oldTeams.form.policiesTooltip")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="policies"
                    className="mt-8"
                    help={t("oldTeams.form.policiesHelp")}
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      placeholder={t("oldTeams.form.policiesPlaceholder")}
                      options={policiesList.map((name) => ({
                        value: name,
                        label: name,
                      }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("oldTeams.form.accessGroups")}{" "}
                        <Tooltip title={t("oldTeams.form.accessGroupsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="access_group_ids"
                    className="mt-8"
                    help={t("oldTeams.form.accessGroupsHelp")}
                  >
                    <AccessGroupSelector placeholder={t("oldTeams.form.accessGroupsPlaceholder")} />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("oldTeams.form.allowedVectorStores")}{" "}
                        <Tooltip title={t("oldTeams.form.allowedVectorStoresTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="allowed_vector_store_ids"
                    className="mt-8"
                    help={t("oldTeams.form.allowedVectorStoresHelp")}
                  >
                    <VectorStoreSelector
                      onChange={(values: string[]) => form.setFieldValue("allowed_vector_store_ids", values)}
                      value={form.getFieldValue("allowed_vector_store_ids")}
                      accessToken={accessToken || ""}
                      placeholder={t("oldTeams.form.allowedVectorStoresPlaceholder")}
                    />
                  </Form.Item>
                  <Form.Item
                    label={t("oldTeams.form.allowedPassThroughRoutes")}
                    name="allowed_passthrough_routes"
                    className="mt-8"
                  >
                    <Tooltip
                      title={
                        !premiumUser
                          ? t("oldTeams.form.allowedPassThroughRoutesPremiumTooltip")
                          : !isProxyAdminRole(userRole || "")
                            ? t("oldTeams.form.allowedPassThroughRoutesAdminTooltip")
                            : ""
                      }
                      placement="top"
                    >
                      <PassThroughRoutesSelector
                        onChange={(values: string[]) => form.setFieldValue("allowed_passthrough_routes", values)}
                        value={form.getFieldValue("allowed_passthrough_routes")}
                        accessToken={accessToken || ""}
                        placeholder={t("oldTeams.form.allowedPassThroughRoutesPlaceholder")}
                        disabled={!premiumUser || !isProxyAdminRole(userRole || "")}
                      />
                    </Tooltip>
                  </Form.Item>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("oldTeams.accordions.mcpSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item
                    label={
                      <span>
                        {t("oldTeams.form.allowedMcpServers")}{" "}
                        <Tooltip title={t("oldTeams.form.allowedMcpServersTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="allowed_mcp_servers_and_groups"
                    className="mt-4"
                    help={t("oldTeams.form.allowedMcpServersHelp")}
                  >
                    <MCPServerSelector
                      onChange={(val: any) => form.setFieldValue("allowed_mcp_servers_and_groups", val)}
                      value={form.getFieldValue("allowed_mcp_servers_and_groups")}
                      accessToken={accessToken || ""}
                      placeholder={t("oldTeams.form.allowedMcpServersPlaceholder")}
                    />
                  </Form.Item>

                  {/* Hidden field to register mcp_tool_permissions with the form */}
                  <Form.Item name="mcp_tool_permissions" initialValue={{}} hidden>
                    <Input type="hidden" />
                  </Form.Item>

                  <Form.Item
                    noStyle
                    shouldUpdate={(prevValues, currentValues) =>
                      prevValues.allowed_mcp_servers_and_groups !== currentValues.allowed_mcp_servers_and_groups ||
                      prevValues.mcp_tool_permissions !== currentValues.mcp_tool_permissions
                    }
                  >
                    {() => (
                      <div className="mt-6">
                        <MCPToolPermissions
                          accessToken={accessToken || ""}
                          selectedServers={form.getFieldValue("allowed_mcp_servers_and_groups")?.servers || []}
                          toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
                          onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
                        />
                      </div>
                    )}
                  </Form.Item>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("oldTeams.accordions.agentSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item
                    label={
                      <span>
                        {t("oldTeams.form.allowedAgents")}{" "}
                        <Tooltip title={t("oldTeams.form.allowedAgentsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="allowed_agents_and_groups"
                    className="mt-4"
                    help={t("oldTeams.form.allowedAgentsHelp")}
                  >
                    <AgentSelector
                      onChange={(val: any) => form.setFieldValue("allowed_agents_and_groups", val)}
                      value={form.getFieldValue("allowed_agents_and_groups")}
                      accessToken={accessToken || ""}
                      placeholder={t("oldTeams.form.allowedAgentsPlaceholder")}
                    />
                  </Form.Item>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("oldTeams.accordions.searchToolSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item
                    label={
                      <span>
                        {t("oldTeams.form.allowedSearchTools")}{" "}
                        <Tooltip title={t("oldTeams.form.allowedSearchToolsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="object_permission_search_tools"
                    className="mt-4"
                    help={t("oldTeams.form.allowedSearchToolsHelp")}
                  >
                    <SearchToolSelector
                      onChange={(vals: string[]) => form.setFieldValue("object_permission_search_tools", vals)}
                      value={form.getFieldValue("object_permission_search_tools")}
                      accessToken={accessToken || ""}
                      placeholder={t("oldTeams.form.allowedSearchToolsPlaceholder")}
                    />
                  </Form.Item>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("oldTeams.accordions.loggingSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <div className="mt-4">
                    <PremiumLoggingSettings
                      value={loggingSettings}
                      onChange={setLoggingSettings}
                      premiumUser={premiumUser}
                    />
                  </div>
                </AccordionBody>
              </Accordion>

              <Accordion key={`router-settings-accordion-${routerSettingsKey}`} className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("oldTeams.accordions.routerSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <div className="mt-4 w-full">
                    <RouterSettingsAccordion
                      key={routerSettingsKey}
                      accessToken={accessToken || ""}
                      value={routerSettings || undefined}
                      onChange={setRouterSettings}
                      modelData={
                        userModels.length > 0 ? { data: userModels.map((model) => ({ model_name: model })) } : undefined
                      }
                    />
                  </div>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("oldTeams.accordions.modelAliases")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <div className="mt-4">
                    <Text type="secondary" style={{ fontSize: 14, marginBottom: 16, display: "block" }}>
                      {t("oldTeams.form.modelAliasesDescription")}
                    </Text>
                    <ModelAliasManager
                      accessToken={accessToken || ""}
                      initialModelAliases={modelAliases}
                      onAliasUpdate={setModelAliases}
                      showExampleConfig={false}
                    />
                  </div>
                </AccordionBody>
              </Accordion>
            </>
            <div style={{ textAlign: "right", marginTop: "10px" }}>
              <Button htmlType="submit" data-testid="create-team-submit">
                {t("oldTeams.createTeam")}
              </Button>
            </div>
          </Form>
        </Modal>
      )}
    </Content>
  );
};

export default Teams;
