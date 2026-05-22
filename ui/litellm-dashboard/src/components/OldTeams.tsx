import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import AvailableTeamsPanel from "@/components/team/available_teams";
import TeamInfoView from "@/components/team/TeamInfo";
import TeamSSOSettings from "@/components/TeamSSOSettings";
import { isProxyAdminRole } from "@/utils/roles";
import {
  InfoCircleOutlined,
  PlusOutlined,
  TeamOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Accordion,
  AccordionBody,
  AccordionHeader,
  TextInput,
} from "@tremor/react";
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
import {
  Organization,
  fetchMCPAccessGroups,
  getGuardrailsList,
  getPoliciesList,
  teamDeleteCall,
} from "./networking";
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

  const fetchTeamsV2 = async (opts: {
    page?: number;
    size?: number;
    sortBy?: string;
    sortOrder?: string;
    organizationID?: string;
    search?: string;
  } = {}) => {
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
      const response: TeamsResponse = await v2TeamListCall(
        accessToken,
        page,
        size,
        {
          organizationID: organizationID || null,
          search: search || null,
          userID: userRole !== "Admin" && userRole !== "Admin Viewer" ? userID : null,
          sortBy: sortBy || null,
          sortOrder: sortOrder || null,
        },
      );
      setTeams(response.teams ?? []);
      setTotalTeams(response.total ?? 0);
    } catch (err: any) {
      setFetchError(err?.message || "Failed to fetch teams");
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
      NotificationsManager.success("Team deleted successfully");
    } catch (error) {
      NotificationsManager.fromBackend("Error deleting the team: " + error);
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

        NotificationsManager.info("Creating Team");

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
        NotificationsManager.success("Team created");
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
      NotificationsManager.fromBackend("Error creating the team: " + error);
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
      const response: TeamsResponse = await v2TeamListCall(
        accessToken,
        1,
        pageSize,
        {
          organizationID: newFilters.organization_id || null,
          search: newFilters.search || null,
          userID: userRole !== "Admin" && userRole !== "Admin Viewer" ? userID : null,
          sortBy: newFilters.sort_by || null,
          sortOrder: newFilters.sort_order || null,
        },
      );
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

  const handleTableSort = (_pagination: unknown, _filters: unknown, sorter: SorterResult<Team> | SorterResult<Team>[]) => {
    const s = Array.isArray(sorter) ? sorter[0] : sorter;
    const sortBy = s.order ? (s.columnKey as string) : "created_at";
    const sortOrder = s.order === "ascend" ? "asc" : s.order === "descend" ? "desc" : "desc";
    setFilters((prev) => ({ ...prev, sort_by: sortBy, sort_order: sortOrder }));
    fetchTeamsV2({ sortBy, sortOrder });
  };

  const teamColumns: ColumnsType<Team> = useMemo(() => [
    {
      title: "Team ID",
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
      title: "Team Alias",
      dataIndex: "team_alias",
      key: "team_alias",
      ellipsis: true,
      sorter: true,
      render: (alias: string | undefined) => (
        <Text style={{ fontSize: 14 }}>
          {alias || <Text type="secondary" italic>—</Text>}
        </Text>
      ),
    },
    {
      title: "Organization",
      key: "organization",
      width: 160,
      ellipsis: true,
      render: (_: unknown, record: Team) => {
        const orgAlias = getOrganizationAlias(record.organization_id, organizationsData || organizations);
        return record.organization_id ? <Text ellipsis style={{ fontSize: 14 }}>{orgAlias}</Text> : <Text type="secondary">—</Text>;
      },
    },
    {
      title: "Resources",
      key: "resources",
      width: 240,
      render: (_: unknown, record: Team) => {
        const memberCount = perTeamInfo?.[record.team_id]?.team_info?.members_with_roles?.length ?? 0;
        const modelCount = record.models?.length ?? 0;
        const keyCount = perTeamInfo?.[record.team_id]?.keys?.length ?? 0;
        return (
          <Flex gap={12} align="center">
            <Tooltip title={`${memberCount} Members`}>
              <Tag color="purple" style={{ fontSize: 14, padding: "2px 8px", margin: 0 }}>
                <Flex align="center" gap={6}>
                  <UsersIcon size={14} />
                  {memberCount}
                </Flex>
              </Tag>
            </Tooltip>
            <Tooltip title={`${modelCount} Models`}>
              <Tag color="blue" style={{ fontSize: 14, padding: "2px 8px", margin: 0 }}>
                <Flex align="center" gap={6}>
                  <LayersIcon size={14} />
                  {modelCount}
                </Flex>
              </Tag>
            </Tooltip>
            <Tooltip title={`${keyCount} Keys`}>
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
      title: "Spend / Budget",
      key: "spend",
      width: 200,
      sorter: true,
      render: (_: unknown, record: Team) => {
        const spendVal = record.spend ?? 0;
        const budgetVal = record.max_budget;
        const spendStr = `$${spendVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        const budgetStr = budgetVal != null
          ? `$${budgetVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
          : "Unlimited";
        const percent = budgetVal != null && budgetVal > 0 ? Math.min((spendVal / budgetVal) * 100, 100) : null;
        return (
          <Flex vertical gap={2}>
            <Text style={{ fontSize: 13 }}>
              {spendStr}
              <Text type="secondary" style={{ fontSize: 12 }}>{" / "}{budgetStr}</Text>
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
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 130,
      ellipsis: true,
      sorter: true,
      render: (date: string | undefined) => (
        <Text type="secondary" style={{ fontSize: 13 }}>
          {date ? new Date(date).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "—"}
        </Text>
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 120,
      align: "right" as const,
      render: (_: unknown, record: Team) => (
        <Space size={4}>
          <TableIconActionButton
            variant="Copy"
            tooltipText="Copy Team ID"
            onClick={() => {
              navigator.clipboard.writeText(record.team_id)
                .then(() => message.success("Team ID copied"))
                .catch(() => message.error("Failed to copy"));
            }}
          />
          {userRole === "Admin" && (
            <>
              <TableIconActionButton
                variant="Edit"
                tooltipText="Edit team"
                dataTestId="edit-team-button"
                onClick={() => {
                  setSelectedTeamId(record.team_id);
                  setEditTeam(true);
                }}
              />
              <TableIconActionButton
                variant="Delete"
                tooltipText="Delete team"
                dataTestId="delete-team-button"
                onClick={() => handleDelete(record)}
              />
            </>
          )}
        </Space>
      ),
    },
  ], [userRole, perTeamInfo, organizationsData, organizations]);

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
            Failed to load teams
          </Text>
          <Text type="secondary" style={{ fontSize: 13 }}>
            {fetchError}
          </Text>
          <Button icon={<ReloadOutlined />} onClick={handleRetry}>
            Retry
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
                <Text style={{ fontSize: 15, color: "#595959" }}>No teams yet</Text>
              </div>
              <div style={{ marginTop: 4 }}>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  Create your first team to organize members and manage access to models.
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
                  Create Team
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
      label: "Your Teams",
      children: (
        <>
          <Card styles={{ body: { padding: 0 } }}>
            <Flex
              justify="space-between"
              align="center"
              style={{ padding: "12px 16px" }}
            >
              <Flex gap={12} align="center">
                <Input
                  prefix={<SearchIcon size={16} />}
                  suffix={isSearching ? <AntDLoadingSpinner size="small" /> : null}
                  placeholder="Search teams by name or ID..."
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
                showTotal={(total) => `${total} teams`}
                showSizeChanger
                pageSizeOptions={["10", "20", "50"]}
              />
            </Flex>

            {renderTeamsContent()}
          </Card>

          <DeleteResourceModal
            isOpen={isDeleteModalOpen}
            title="Delete Team?"
            alertMessage={
              teamToDelete?.keys?.length === 0
                ? undefined
                : `Warning: This team has ${teamToDelete?.keys?.length} keys associated with it. Deleting the team will also delete all associated keys. This action is irreversible.`
            }
            message="Are you sure you want to delete this team and all its keys? This action cannot be undone."
            resourceInformationTitle="Team Information"
            resourceInformation={[
              { label: "Team ID", value: teamToDelete?.team_id, code: true },
              { label: "Team Name", value: teamToDelete?.team_alias },
              { label: "Keys", value: teamToDelete?.keys?.length },
              { label: "Members", value: teamToDelete?.members_with_roles?.length },
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
      label: "Available Teams",
      children: <AvailableTeamsPanel accessToken={accessToken} userID={userID} />,
    },
    ...(isProxyAdminRole(userRole || "")
      ? [
          {
            key: "default-settings",
            label: "Default Team Settings",
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
                Teams
              </Title>
              <Text type="secondary">
                Manage teams, members, and their access to models and budgets
              </Text>
            </Space>
            {canCreateOrManageTeams(userRole, userID, organizations) && (
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setIsTeamModalVisible(true)} data-testid="create-team-button">
                Create Team
              </Button>
            )}
          </Flex>

          <Tabs items={tabItems} />
        </>
      )}

      {canCreateOrManageTeams(userRole, userID, organizations) && (
            <Modal
              title="Create Team"
              open={isTeamModalVisible}
              width={1000}
              footer={null}
              onOk={handleOk}
              onCancel={handleCancel}
            >
              <Form
                form={form}
                onFinish={handleCreate}
                labelCol={{ span: 8 }}
                wrapperCol={{ span: 16 }}
                labelAlign="left"
              >
                <>
                  <Form.Item
                    label="Team Name"
                    name="team_alias"
                    rules={[
                      {
                        required: true,
                        message: "Please input a team name",
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
                              Organization{" "}
                              <Tooltip
                                title={
                                  <span>
                                    Organizations can have multiple teams. Learn more about{" "}
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
                                      user management hierarchy
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
                                  message: "Please select an organization",
                                },
                              ]
                              : []
                          }
                          help={
                            isSingleOrg
                              ? "You can only create teams within this organization"
                              : isOrgAdmin
                                ? "required"
                                : ""
                          }
                        >
                          <Select
                            showSearch
                            allowClear={!isOrgAdmin}
                            disabled={isSingleOrg}
                            placeholder={hasNoOrgs ? "No organizations available" : "Search or select an Organization"}
                            onChange={(value) => {
                              form.setFieldValue("organization_id", value);
                              setCurrentOrgForCreateTeam(
                                adminOrgs?.find((org) => org.organization_id === value) || null,
                              );
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
                            <Text style={{ color: "#1e40af", fontSize: 14 }}>
                              Please select an organization to create a team for. You can only create teams within
                              organizations where you are an admin.
                            </Text>
                          </div>
                        )}
                      </>
                    );
                  })()}
                  <Form.Item
                    label={
                      <span>
                        Models{" "}
                        <Tooltip title="These are the models that your selected team has access to">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    rules={[
                      {
                        required: true,
                        message: "Please select at least one model",
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

                  <Form.Item label="Max Budget (USD)" name="max_budget">
                    <NumericalInput step={0.01} precision={2} width={200} />
                  </Form.Item>
                  <Form.Item className="mt-8" label="Reset Budget" name="budget_duration">
                    <Select defaultValue={null} placeholder="n/a">
                      <Select.Option value="24h">daily</Select.Option>
                      <Select.Option value="7d">weekly</Select.Option>
                      <Select.Option value="30d">monthly</Select.Option>
                    </Select>
                  </Form.Item>
                  <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
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
                      <b>Additional Settings</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <Form.Item
                        label="Team ID"
                        name="team_id"
                        help="ID of the team you want to create. If not provided, it will be generated automatically."
                      >
                        <TextInput
                          onChange={(e) => {
                            e.target.value = e.target.value.trim();
                          }}
                        />
                      </Form.Item>
                      <Form.Item
                        label="Team Member Budget (USD)"
                        name="team_member_budget"
                        normalize={(value) => (value ? Number(value) : undefined)}
                        tooltip="This is the individual budget for a user in the team."
                      >
                        <NumericalInput step={0.01} precision={2} width={200} />
                      </Form.Item>
                      <Form.Item
                        label="Team Member Key Duration (eg: 1d, 1mo)"
                        name="team_member_key_duration"
                        tooltip="Set a limit to the duration of a team member's key. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days), 1mo (month)"
                      >
                        <TextInput placeholder="e.g., 30d" />
                      </Form.Item>
                      <Form.Item
                        label="Team Member RPM Limit"
                        name="team_member_rpm_limit"
                        tooltip="The RPM (Requests Per Minute) limit for individual team members"
                      >
                        <NumericalInput step={1} width={400} />
                      </Form.Item>
                      <Form.Item
                        label="Team Member TPM Limit"
                        name="team_member_tpm_limit"
                        tooltip="The TPM (Tokens Per Minute) limit for individual team members"
                      >
                        <NumericalInput step={1} width={400} />
                      </Form.Item>
                      <Form.Item
                        label="Metadata"
                        name="metadata"
                        help="Additional team metadata. Enter metadata as JSON object."
                      >
                        <Input.TextArea rows={4} />
                      </Form.Item>
                      <Form.Item
                        label="Secret Manager Settings"
                        name="secret_manager_settings"
                        help={
                          premiumUser
                            ? "Enter secret manager configuration as a JSON object."
                            : "Premium feature - Upgrade to manage secret manager settings."
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
                                return Promise.reject(new Error("Please enter valid JSON"));
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
                            Guardrails{" "}
                            <Tooltip title="Setup your first guardrail">
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
                        help="Select existing guardrails or enter new ones"
                      >
                        <Select
                          mode="tags"
                          style={{ width: "100%" }}
                          placeholder="Select or enter guardrails"
                          options={guardrailsList.map((name) => ({
                            value: name,
                            label: name,
                          }))}
                        />
                      </Form.Item>
                      <Form.Item
                        label={
                          <span>
                            Disable Global Guardrails{" "}
                            <Tooltip title="When enabled, this team will bypass any guardrails configured to run on every request (global guardrails)">
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="disable_global_guardrails"
                        className="mt-4"
                        valuePropName="checked"
                        help="Bypass global guardrails for this team"
                      >
                        <Switch
                          disabled={!premiumUser}
                          checkedChildren={
                            premiumUser ? "Yes" : "Premium feature - Upgrade to disable global guardrails by team"
                          }
                          unCheckedChildren={
                            premiumUser ? "No" : "Premium feature - Upgrade to disable global guardrails by team"
                          }
                        />
                      </Form.Item>
                      <Form.Item
                        label={
                          <span>
                            Policies{" "}
                            <Tooltip title="Apply policies to this team to control guardrails and other settings">
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
                        help="Select existing policies or enter new ones"
                      >
                        <Select
                          mode="tags"
                          style={{ width: "100%" }}
                          placeholder="Select or enter policies"
                          options={policiesList.map((name) => ({
                            value: name,
                            label: name,
                          }))}
                        />
                      </Form.Item>
                      <Form.Item
                        label={
                          <span>
                            Access Groups{" "}
                            <Tooltip title="Assign access groups to this team. Access groups control which models, MCP servers, and agents this team can use">
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="access_group_ids"
                        className="mt-8"
                        help="Select access groups to assign to this team"
                      >
                        <AccessGroupSelector placeholder="Select access groups (optional)" />
                      </Form.Item>
                      <Form.Item
                        label={
                          <span>
                            Allowed Vector Stores{" "}
                            <Tooltip title="Select which vector stores this team can access by default. Leave empty for access to all vector stores">
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="allowed_vector_store_ids"
                        className="mt-8"
                        help="Select vector stores this team can access. Leave empty for access to all vector stores"
                      >
                        <VectorStoreSelector
                          onChange={(values: string[]) => form.setFieldValue("allowed_vector_store_ids", values)}
                          value={form.getFieldValue("allowed_vector_store_ids")}
                          accessToken={accessToken || ""}
                          placeholder="Select vector stores (optional)"
                        />
                      </Form.Item>
                    </AccordionBody>
                  </Accordion>

                  <Accordion className="mt-8 mb-8">
                    <AccordionHeader>
                      <b>MCP Settings</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <Form.Item
                        label={
                          <span>
                            Allowed MCP Servers{" "}
                            <Tooltip title="Select which MCP servers or access groups this team can access">
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="allowed_mcp_servers_and_groups"
                        className="mt-4"
                        help="Select MCP servers or access groups this team can access"
                      >
                        <MCPServerSelector
                          onChange={(val: any) => form.setFieldValue("allowed_mcp_servers_and_groups", val)}
                          value={form.getFieldValue("allowed_mcp_servers_and_groups")}
                          accessToken={accessToken || ""}
                          placeholder="Select MCP servers or access groups (optional)"
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
                      <b>Agent Settings</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <Form.Item
                        label={
                          <span>
                            Allowed Agents{" "}
                            <Tooltip title="Select which agents or access groups this team can access">
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="allowed_agents_and_groups"
                        className="mt-4"
                        help="Select agents or access groups this team can access"
                      >
                        <AgentSelector
                          onChange={(val: any) => form.setFieldValue("allowed_agents_and_groups", val)}
                          value={form.getFieldValue("allowed_agents_and_groups")}
                          accessToken={accessToken || ""}
                          placeholder="Select agents or access groups (optional)"
                        />
                      </Form.Item>
                    </AccordionBody>
                  </Accordion>

                  <Accordion className="mt-8 mb-8">
                    <AccordionHeader>
                      <b>Search Tool Settings</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <Form.Item
                        label={
                          <span>
                            Allowed Search Tools{" "}
                            <Tooltip title="Select which search tools this team can access. Leave empty to allow all search tools.">
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="object_permission_search_tools"
                        className="mt-4"
                        help="Restrict which configured search tools keys on this team may call."
                      >
                        <SearchToolSelector
                          onChange={(vals: string[]) => form.setFieldValue("object_permission_search_tools", vals)}
                          value={form.getFieldValue("object_permission_search_tools")}
                          accessToken={accessToken || ""}
                          placeholder="Select search tools (optional, empty = all allowed)"
                        />
                      </Form.Item>
                    </AccordionBody>
                  </Accordion>

                  <Accordion className="mt-8 mb-8">
                    <AccordionHeader>
                      <b>Logging Settings</b>
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
                      <b>Router Settings</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <div className="mt-4 w-full">
                        <RouterSettingsAccordion
                          key={routerSettingsKey}
                          accessToken={accessToken || ""}
                          value={routerSettings || undefined}
                          onChange={setRouterSettings}
                          modelData={userModels.length > 0 ? { data: userModels.map((model) => ({ model_name: model })) } : undefined}
                        />
                      </div>
                    </AccordionBody>
                  </Accordion>

                  <Accordion className="mt-8 mb-8">
                    <AccordionHeader>
                      <b>Model Aliases</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <div className="mt-4">
                        <Text type="secondary" style={{ fontSize: 14, marginBottom: 16, display: "block" }}>
                          Create custom aliases for models that can be used by team members in API calls. This allows
                          you to create shortcuts for specific models.
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
                  <Button htmlType="submit" data-testid="create-team-submit">Create Team</Button>
                </div>
              </Form>
            </Modal>
          )}
    </Content>
  );
};

export default Teams;
