import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import AvailableTeamsPanel from "@/components/team/available_teams";
import TeamInfoView from "@/components/team/TeamInfo";
import TeamSSOSettings from "@/components/TeamSSOSettings";
import { isProxyAdminRole } from "@/utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import { ChevronDownIcon, ChevronRightIcon, RefreshIcon } from "@heroicons/react/outline";
import { FilterInput } from "@/components/common_components/Filters/FilterInput";
import { FiltersButton } from "@/components/common_components/Filters/FiltersButton";
import { ResetFiltersButton } from "@/components/common_components/Filters/ResetFiltersButton";
import { Search, User } from "lucide-react";
import {
  Accordion,
  AccordionBody,
  AccordionHeader,
  Badge,
  Button,
  Card,
  Col,
  Grid,
  Icon,
  Select,
  SelectItem,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
  TextInput,
} from "@tremor/react";
import { Button as Button2, Form, Input, Modal, Select as Select2, Switch, Tooltip, Typography } from "antd";
import React, { useEffect, useState } from "react";
import { formatNumberWithCommas } from "../utils/dataUtils";
import AccessGroupSelector from "./common_components/AccessGroupSelector";
import AgentSelector from "./agent_management/AgentSelector";
import { fetchTeams } from "./common_components/fetch_teams";
import ModelAliasManager from "./common_components/ModelAliasManager";
import PremiumLoggingSettings from "./common_components/PremiumLoggingSettings";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "./common_components/RouterSettingsAccordion";
import {
  fetchAvailableModelsForTeamOrKey,
  getModelDisplayName,
  unfurlWildcardModelsInList,
} from "./key_team_helpers/fetch_available_models_team_key";
import type { KeyResponse, Team } from "./key_team_helpers/key_list";
import MCPServerSelector from "./mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "./mcp_server_management/MCPToolPermissions";
import NotificationsManager from "./molecules/notifications_manager";
import { Organization, fetchMCPAccessGroups, getGuardrailsList, getPoliciesList, teamDeleteCall } from "./networking";
import NumericalInput from "./shared/numerical_input";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";

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
  team_id: string;
  team_alias: string;
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
import TableIconActionButton from "./common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { Member, teamCreateCall, v2TeamListCall } from "./networking";
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
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [currentOrg, setCurrentOrg] = useState<Organization | null>(null);
  const [currentOrgForCreateTeam, setCurrentOrgForCreateTeam] = useState<Organization | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    team_id: "",
    team_alias: "",
    organization_id: "",
    sort_by: "created_at",
    sort_order: "desc",
  });

  useEffect(() => {
    console.log(`inside useeffect - ${lastRefreshed}`);
    if (accessToken) {
      // Call your function here
      fetchTeams(accessToken, userID, userRole, currentOrg, setTeams);
    }
    handleRefreshClick();
  }, [lastRefreshed]);

  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();
  const { Title, Paragraph } = Typography;
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
  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});
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
      await fetchTeams(accessToken, userID, userRole, currentOrg, setTeams);
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

        // Transform allowed_vector_store_ids and allowed_mcp_servers_and_groups into object_permission
        if (
          (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) ||
          (formValues.allowed_mcp_servers_and_groups &&
            (formValues.allowed_mcp_servers_and_groups.servers?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.accessGroups?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.toolPermissions))
        ) {
          formValues.object_permission = {};
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

          // Add tool permissions separately
          if (formValues.mcp_tool_permissions && Object.keys(formValues.mcp_tool_permissions).length > 0) {
            if (!formValues.object_permission) {
              formValues.object_permission = {};
            }
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

        const response: any = await teamCreateCall(accessToken, formValues);
        if (teams !== null) {
          setTeams([...teams, response]);
        } else {
          setTeams([response]);
        }
        console.log(`response for team create call: ${response}`);
        NotificationsManager.success("Team created");
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

  const handleRefreshClick = () => {
    // Update the 'lastRefreshed' state to the current date and time
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  };

  const handleFilterChange = (key: keyof FilterState, value: string) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    // Call teamListCall with the new filters
    if (accessToken) {
      v2TeamListCall(
        accessToken,
        newFilters.organization_id || null,
        null,
        newFilters.team_id || null,
        newFilters.team_alias || null,
      )
        .then((response) => {
          if (response && response.teams) {
            setTeams(response.teams);
          }
        })
        .catch((error) => {
          console.error("Error fetching teams:", error);
        });
    }
  };

  const handleSortChange = (sortBy: string, sortOrder: "asc" | "desc") => {
    const newFilters = {
      ...filters,
      sort_by: sortBy,
      sort_order: sortOrder,
    };
    setFilters(newFilters);
    // Call teamListCall with the new sort parameters
    if (accessToken) {
      v2TeamListCall(
        accessToken,
        filters.organization_id || null,
        null,
        filters.team_id || null,
        filters.team_alias || null,
      )
        .then((response) => {
          if (response && response.teams) {
            setTeams(response.teams);
          }
        })
        .catch((error) => {
          console.error("Error fetching teams:", error);
        });
    }
  };

  const handleFilterReset = () => {
    setFilters({
      team_id: "",
      team_alias: "",
      organization_id: "",
      sort_by: "created_at",
      sort_order: "desc",
    });
    // Reset teams list
    if (accessToken) {
      v2TeamListCall(accessToken, null, userID || null, null, null)
        .then((response) => {
          if (response && response.teams) {
            setTeams(response.teams);
          }
        })
        .catch((error) => {
          console.error("Error fetching teams:", error);
        });
    }
  };

  return (
    <div className="w-full mx-4 h-[75vh]">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
        <Col numColSpan={1} className="flex flex-col gap-2">
          {canCreateOrManageTeams(userRole, userID, organizations) && (
            <Button className="w-fit" onClick={() => setIsTeamModalVisible(true)}>
              + Create New Team
            </Button>
          )}
          {selectedTeamId ? (
            <TeamInfoView
              teamId={selectedTeamId}
              onUpdate={(data) => {
                setTeams((teams) => {
                  if (teams == null) {
                    return teams;
                  }
                  const updated = teams.map((team) => {
                    if (data.team_id === team.team_id) {
                      return updateExistingKeys(team, data);
                    }
                    return team;
                  });
                  // Minimal fix: refresh the full team list after an update
                  if (accessToken) {
                    fetchTeams(accessToken, userID, userRole, currentOrg, setTeams);
                  }
                  return updated;
                });
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
            <TabGroup className="gap-2 h-[75vh] w-full">
              <TabList className="flex justify-between mt-2 w-full items-center">
                <div className="flex">
                  <Tab>Your Teams</Tab>
                  <Tab>Available Teams</Tab>
                  {isProxyAdminRole(userRole || "") && <Tab>Default Team Settings</Tab>}
                </div>
                <div className="flex items-center space-x-2">
                  {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
                  <Icon
                    icon={RefreshIcon} // Modify as necessary for correct icon name
                    variant="shadow"
                    size="xs"
                    className="self-center"
                    onClick={handleRefreshClick}
                  />
                </div>
              </TabList>
              <TabPanels>
                <TabPanel>
                  <Text>
                    Click on &ldquo;Team ID&rdquo; to view team details <b>and</b> manage team members.
                  </Text>
                  <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
                    <Col numColSpan={1}>
                      <Card className="w-full mx-auto flex-auto overflow-hidden overflow-y-auto max-h-[50vh]">
                        <div className="border-b px-6 py-4">
                          <div className="flex flex-col space-y-4">
                            {/* Search and Filter Controls */}
                            <div className="flex flex-wrap items-center gap-3">
                              {/* Team Alias Search */}
                              <FilterInput
                                placeholder="Search by Team Name..."
                                value={filters.team_alias}
                                onChange={(value) => handleFilterChange("team_alias", value)}
                                icon={Search}
                              />

                              {/* Filter Button */}
                              <FiltersButton
                                onClick={() => setShowFilters(!showFilters)}
                                active={showFilters}
                                hasActiveFilters={!!(filters.team_id || filters.team_alias || filters.organization_id)}
                              />

                              {/* Reset Filters Button */}
                              <ResetFiltersButton onClick={handleFilterReset} />
                            </div>

                            {/* Additional Filters */}
                            {showFilters && (
                              <div className="flex flex-wrap items-center gap-3 mt-3">
                                {/* Team ID Search */}
                                <FilterInput
                                  placeholder="Enter Team ID"
                                  value={filters.team_id}
                                  onChange={(value) => handleFilterChange("team_id", value)}
                                  icon={User}
                                />

                                {/* Organization Dropdown */}
                                <div className="w-64">
                                  <Select
                                    value={filters.organization_id || ""}
                                    onValueChange={(value) => handleFilterChange("organization_id", value)}
                                    placeholder="Select Organization"
                                  >
                                    {organizations?.map((org) => (
                                      <SelectItem key={org.organization_id} value={org.organization_id || ""}>
                                        {org.organization_alias || org.organization_id}
                                      </SelectItem>
                                    ))}
                                  </Select>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableHeaderCell>Team Name</TableHeaderCell>
                              <TableHeaderCell>Team ID</TableHeaderCell>
                              <TableHeaderCell>Created</TableHeaderCell>
                              <TableHeaderCell>Spend (USD)</TableHeaderCell>
                              <TableHeaderCell>Budget (USD)</TableHeaderCell>
                              <TableHeaderCell>Models</TableHeaderCell>
                              <TableHeaderCell>Organization</TableHeaderCell>
                              <TableHeaderCell>Info</TableHeaderCell>
                              <TableHeaderCell>Actions</TableHeaderCell>
                            </TableRow>
                          </TableHead>

                          <TableBody>
                            {teams && teams.length > 0 ? (
                              teams
                                .filter((team) => {
                                  if (!currentOrg) return true;
                                  return team.organization_id === currentOrg.organization_id;
                                })
                                .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
                                .map((team: any) => (
                                  <TableRow key={team.team_id}>
                                    <TableCell
                                      style={{
                                        maxWidth: "4px",
                                        whiteSpace: "pre-wrap",
                                        overflow: "hidden",
                                      }}
                                    >
                                      {team["team_alias"]}
                                    </TableCell>
                                    <TableCell>
                                      <div className="overflow-hidden">
                                        <Tooltip title={team.team_id}>
                                          <Button
                                            size="xs"
                                            variant="light"
                                            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
                                            onClick={() => {
                                              // Add click handler
                                              setSelectedTeamId(team.team_id);
                                            }}
                                          >
                                            {team.team_id.slice(0, 7)}...
                                          </Button>
                                        </Tooltip>
                                      </div>
                                    </TableCell>
                                    <TableCell
                                      style={{
                                        maxWidth: "4px",
                                        whiteSpace: "pre-wrap",
                                        overflow: "hidden",
                                      }}
                                    >
                                      {team.created_at ? new Date(team.created_at).toLocaleDateString() : "N/A"}
                                    </TableCell>
                                    <TableCell
                                      style={{
                                        maxWidth: "4px",
                                        whiteSpace: "pre-wrap",
                                        overflow: "hidden",
                                      }}
                                    >
                                      {formatNumberWithCommas(team["spend"], 4)}
                                    </TableCell>
                                    <TableCell
                                      style={{
                                        maxWidth: "4px",
                                        whiteSpace: "pre-wrap",
                                        overflow: "hidden",
                                      }}
                                    >
                                      {team["max_budget"] !== null && team["max_budget"] !== undefined
                                        ? team["max_budget"]
                                        : "No limit"}
                                    </TableCell>
                                    <TableCell
                                      style={{
                                        maxWidth: "8-x",
                                        whiteSpace: "pre-wrap",
                                        overflow: "hidden",
                                      }}
                                      className={team.models.length > 3 ? "px-0" : ""}
                                    >
                                      <div className="flex flex-col">
                                        {Array.isArray(team.models) ? (
                                          <div className="flex flex-col">
                                            {team.models.length === 0 ? (
                                              <Badge size={"xs"} className="mb-1" color="red">
                                                <Text>All Proxy Models</Text>
                                              </Badge>
                                            ) : (
                                              <>
                                                <div className="flex items-start">
                                                  {team.models.length > 3 && (
                                                    <div>
                                                      <Icon
                                                        icon={
                                                          expandedAccordions[team.team_id]
                                                            ? ChevronDownIcon
                                                            : ChevronRightIcon
                                                        }
                                                        className="cursor-pointer"
                                                        size="xs"
                                                        onClick={() => {
                                                          setExpandedAccordions((prev) => ({
                                                            ...prev,
                                                            [team.team_id]: !prev[team.team_id],
                                                          }));
                                                        }}
                                                      />
                                                    </div>
                                                  )}
                                                  <div className="flex flex-wrap gap-1">
                                                    {team.models.slice(0, 3).map((model: string, index: number) =>
                                                      model === "all-proxy-models" ? (
                                                        <Badge key={index} size={"xs"} color="red">
                                                          <Text>All Proxy Models</Text>
                                                        </Badge>
                                                      ) : (
                                                        <Badge key={index} size={"xs"} color="blue">
                                                          <Text>
                                                            {model.length > 30
                                                              ? `${getModelDisplayName(model).slice(0, 30)}...`
                                                              : getModelDisplayName(model)}
                                                          </Text>
                                                        </Badge>
                                                      ),
                                                    )}
                                                    {team.models.length > 3 && !expandedAccordions[team.team_id] && (
                                                      <Badge size={"xs"} color="gray" className="cursor-pointer">
                                                        <Text>
                                                          +{team.models.length - 3}{" "}
                                                          {team.models.length - 3 === 1 ? "more model" : "more models"}
                                                        </Text>
                                                      </Badge>
                                                    )}
                                                    {expandedAccordions[team.team_id] && (
                                                      <div className="flex flex-wrap gap-1">
                                                        {team.models.slice(3).map((model: string, index: number) =>
                                                          model === "all-proxy-models" ? (
                                                            <Badge key={index + 3} size={"xs"} color="red">
                                                              <Text>All Proxy Models</Text>
                                                            </Badge>
                                                          ) : (
                                                            <Badge key={index + 3} size={"xs"} color="blue">
                                                              <Text>
                                                                {model.length > 30
                                                                  ? `${getModelDisplayName(model).slice(0, 30)}...`
                                                                  : getModelDisplayName(model)}
                                                              </Text>
                                                            </Badge>
                                                          ),
                                                        )}
                                                      </div>
                                                    )}
                                                  </div>
                                                </div>
                                              </>
                                            )}
                                          </div>
                                        ) : null}
                                      </div>
                                    </TableCell>

                                    <TableCell>
                                      {getOrganizationAlias(team.organization_id, organizationsData || organizations)}
                                    </TableCell>
                                    <TableCell>
                                      <Text>
                                        {perTeamInfo &&
                                          team.team_id &&
                                          perTeamInfo[team.team_id] &&
                                          perTeamInfo[team.team_id].keys &&
                                          perTeamInfo[team.team_id].keys.length}{" "}
                                        Keys
                                      </Text>
                                      <Text>
                                        {perTeamInfo &&
                                          team.team_id &&
                                          perTeamInfo[team.team_id] &&
                                          perTeamInfo[team.team_id].team_info &&
                                          perTeamInfo[team.team_id].team_info.members_with_roles &&
                                          perTeamInfo[team.team_id].team_info.members_with_roles.length}{" "}
                                        Members
                                      </Text>
                                    </TableCell>
                                    <TableCell>
                                      {userRole == "Admin" ? (
                                        <>
                                          <TableIconActionButton
                                            variant="Edit"
                                            onClick={() => {
                                              setSelectedTeamId(team.team_id);
                                              setEditTeam(true);
                                            }}
                                            dataTestId="edit-team-button"
                                            tooltipText="Edit team"
                                          />
                                          <TableIconActionButton
                                            variant="Delete"
                                            onClick={() => handleDelete(team)}
                                            dataTestId="delete-team-button"
                                            tooltipText="Delete team"
                                          />
                                        </>
                                      ) : null}
                                    </TableCell>
                                  </TableRow>
                                ))
                            ) : (
                              <TableRow>
                                <TableCell colSpan={9} className="text-center">
                                  <div className="flex flex-col items-center justify-center py-4">
                                    <Text className="text-lg font-medium mb-2">No teams found</Text>
                                    <Text className="text-sm">Adjust your filters or create a new team</Text>
                                  </div>
                                </TableCell>
                              </TableRow>
                            )}
                          </TableBody>
                        </Table>
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
                      </Card>
                    </Col>
                  </Grid>
                </TabPanel>
                <TabPanel>
                  <AvailableTeamsPanel accessToken={accessToken} userID={userID} />
                </TabPanel>
                {isProxyAdminRole(userRole || "") && (
                  <TabPanel>
                    <TeamSSOSettings accessToken={accessToken} userID={userID || ""} userRole={userRole || ""} />
                  </TabPanel>
                )}
              </TabPanels>
            </TabGroup>
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
                    <TextInput placeholder="" />
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
                          <Select2
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
                              <Select2.Option key={org.organization_id} value={org.organization_id}>
                                <span className="font-medium">{org.organization_alias}</span>{" "}
                                <span className="text-gray-500">({org.organization_id})</span>
                              </Select2.Option>
                            ))}
                          </Select2>
                        </Form.Item>

                        {/* Show message when org admin needs to select organization */}
                        {isOrgAdmin && !isSingleOrg && adminOrgs.length > 1 && (
                          <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-md">
                            <Text className="text-blue-800 text-sm">
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
                    <Select2 defaultValue={null} placeholder="n/a">
                      <Select2.Option value="24h">daily</Select2.Option>
                      <Select2.Option value="7d">weekly</Select2.Option>
                      <Select2.Option value="30d">monthly</Select2.Option>
                    </Select2>
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
                        <Select2
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
                        <Select2
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
                        <Text className="text-sm text-gray-600 mb-4">
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
                  <Button2 htmlType="submit">Create Team</Button2>
                </div>
              </Form>
            </Modal>
          )}
        </Col>
      </Grid>
    </div>
  );
};

export default Teams;
