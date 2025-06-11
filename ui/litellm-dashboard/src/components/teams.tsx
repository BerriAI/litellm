import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Typography } from "antd";
import { teamDeleteCall, teamUpdateCall, teamInfoCall, Organization, DEFAULT_ORGANIZATION } from "./networking";
import TeamMemberModal from "@/components/team/edit_membership";
import { fetchTeams } from "./common_components/fetch_teams";
import {
  InformationCircleIcon,
  PencilAltIcon,
  PencilIcon,
  RefreshIcon,
  StatusOnlineIcon,
  TrashIcon,
  ChevronDownIcon,
  ChevronRightIcon
} from "@heroicons/react/outline";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  message,
  Tooltip
} from "antd";
import NumericalInput from "./shared/numerical_input";
import { fetchAvailableModelsForTeamOrKey, getModelDisplayName, unfurlWildcardModelsInList } from "./key_team_helpers/fetch_available_models_team_key";
import { Select, SelectItem } from "@tremor/react";
import { InfoCircleOutlined } from '@ant-design/icons';
import { getGuardrailsList } from "./networking";
import TeamInfoView, { TeamData } from "@/components/team/team_info";
import TeamSSOSettings from "@/components/TeamSSOSettings";
import { isAdminRole } from "@/utils/roles";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  TextInput,
  Card,
  Icon,
  Button,
  Badge,
  Col,
  Text,
  Grid,
  Accordion,
  AccordionHeader,
  AccordionBody,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Tab
} from "@tremor/react";
import { CogIcon } from "@heroicons/react/outline";
import AvailableTeamsPanel from "@/components/team/available_teams";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import PremiumVectorStoreSelector from "./common_components/PremiumVectorStoreSelector";
import type { KeyResponse, Team } from "./key_team_helpers/key_list";

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
  sort_order: 'asc' | 'desc';
}

interface EditTeamModalProps {
  visible: boolean;
  onCancel: () => void;
  team: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
}

import {
  teamCreateCall,
  teamMemberAddCall,
  teamMemberUpdateCall,
  Member,
  modelAvailableCall,
  v2TeamListCall
} from "./networking";
import { updateExistingKeys } from "@/utils/dataUtils";

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
}

const Teams: React.FC<TeamProps> = ({
  teams,
  searchParams,
  accessToken,
  setTeams,
  userID,
  userRole,
  organizations,
  premiumUser = false
}) => {
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [currentOrg, setCurrentOrg] = useState<Organization | null>(null);
  const [currentOrgForCreateTeam, setCurrentOrgForCreateTeam] = useState<Organization | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    team_id: "",
    team_alias: "",
    organization_id: "",
    sort_by: "created_at",
    sort_order: "desc"
  });

  useEffect(() => {
    console.log(`inside useeffect - ${lastRefreshed}`)
    if (accessToken) {
      // Call your function here
      fetchTeams(accessToken, userID, userRole, currentOrg, setTeams)
    }
    handleRefreshClick()
  }, [lastRefreshed]);

  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();
  const { Title, Paragraph } = Typography;
  const [value, setValue] = useState("");
  const [editModalVisible, setEditModalVisible] = useState(false);

  const [selectedTeam, setSelectedTeam] = useState<null | any>(
    null
  );
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [editTeam, setEditTeam] = useState<boolean>(false);

  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [teamToDelete, setTeamToDelete] = useState<string | null>(null);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [perTeamInfo, setPerTeamInfo] = useState<Record<string, PerTeamInfo>>({});

  // Add this state near the other useState declarations
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});

  useEffect(() => {
    console.log(`currentOrgForCreateTeam: ${currentOrgForCreateTeam}`);
    const models = getOrganizationModels(currentOrgForCreateTeam, userModels);
    console.log(`models: ${models}`);
    setModelsToPick(models);
    form.setFieldValue('models', []);
  }, [currentOrgForCreateTeam, userModels]);

  // Add this useEffect to fetch guardrails
  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        if (accessToken == null) {
          return;
        }

        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map(
          (g: { guardrail_name: string }) => g.guardrail_name
        );
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    fetchGuardrails();
  }, [accessToken]);

  useEffect(() => {
    const fetchTeamInfo = () => {
      if (!teams) return;
      
      const newPerTeamInfo = teams.reduce((acc, team) => {
        acc[team.team_id] = {
          keys: team.keys || [],
          team_info: {
            members_with_roles: team.members_with_roles || []
          }
        };
        return acc;
      }, {} as Record<string, PerTeamInfo>);
      
      setPerTeamInfo(newPerTeamInfo);
    };

    fetchTeamInfo();
  }, [teams]);

  const handleOk = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
  };

  const handleMemberOk = () => {
    setIsAddMemberModalVisible(false);
    setIsEditMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleCancel = () => {
    setIsTeamModalVisible(false);

    form.resetFields();
  };

  const handleMemberCancel = () => {
    setIsAddMemberModalVisible(false);
    setIsEditMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleDelete = async (team_id: string) => {
    // Set the team to delete and open the confirmation modal
    setTeamToDelete(team_id);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (teamToDelete == null || teams == null || accessToken == null) {
      return;
    }

    try {
      await teamDeleteCall(accessToken, teamToDelete);
      // Successfully completed the deletion. Update the state to trigger a rerender.
      fetchTeams(accessToken, userID, userRole, currentOrg, setTeams)
    } catch (error) {
      console.error("Error deleting the team:", error);
      // Handle any error situations, such as displaying an error message to the user.
    }

    // Close the confirmation modal and reset the teamToDelete
    setIsDeleteModalOpen(false);
    setTeamToDelete(null);
  };

  const cancelDelete = () => {
    // Close the confirmation modal and reset the teamToDelete
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
        if (organizationId === "" || typeof organizationId !== 'string') {
          formValues.organization_id = null;
        } else {
          formValues.organization_id = organizationId.trim();
        }

        
        // Remove guardrails from top level since it's now in metadata
        if (existingTeamAliases.includes(newTeamAlias)) {
          throw new Error(
            `Team alias ${newTeamAlias} already exists, please pick another alias`
          );
        }

        message.info("Creating Team");
        // Transform allowed_vector_store_ids into object_permission
        if (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) {
          formValues.object_permission = {
            vector_stores: formValues.allowed_vector_store_ids
          };
          delete formValues.allowed_vector_store_ids;
        }
        const response: any = await teamCreateCall(accessToken, formValues);
        if (teams !== null) {
          setTeams([...teams, response]);
        } else {
          setTeams([response]);
        }
        console.log(`response for team create call: ${response}`);
        message.success("Team created");
        form.resetFields();
        setIsTeamModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the team:", error);
      message.error("Error creating the team: " + error, 20);
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
  }



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
        newFilters.team_alias || null
      ).then((response) => {
        if (response && response.teams) {
          setTeams(response.teams);
        }
      }).catch((error) => {
        console.error("Error fetching teams:", error);
      });
    }
  };

  const handleSortChange = (sortBy: string, sortOrder: 'asc' | 'desc') => {
    const newFilters = {
      ...filters,
      sort_by: sortBy,
      sort_order: sortOrder
    };
    setFilters(newFilters);
    // Call teamListCall with the new sort parameters
    if (accessToken) {
      v2TeamListCall(
        accessToken,
        filters.organization_id || null,
        null,
        filters.team_id || null,
        filters.team_alias || null
      ).then((response) => {
        if (response && response.teams) {
          setTeams(response.teams);
        }
      }).catch((error) => {
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
      sort_order: "desc"
    });
    // Reset teams list
    if (accessToken) {
      v2TeamListCall(accessToken, null, userID || null, null, null).then((response) => {
        if (response && response.teams) {
          setTeams(response.teams);
        }
      }).catch((error) => {
        console.error("Error fetching teams:", error);
      });
    }
  };

  return (
    <div className="w-full mx-4 h-[75vh]">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
        <Col numColSpan={1} className="flex flex-col gap-2">
        {
          (userRole == "Admin" || userRole == "Org Admin") && !selectedTeamId? 
          <Button
            className="w-fit"
            onClick={() => setIsTeamModalVisible(true)}
          >
            + Create New Team
          </Button> : null
        }          
          {selectedTeamId ? (
            <TeamInfoView 
            teamId={selectedTeamId} 
            onUpdate={(data) => {
                setTeams(teams => {
                  if (teams == null) {
                    return teams;
                  }
                
                  return teams.map(team => {
                    if (data.team_id === team.team_id) {
                      return updateExistingKeys(team, data)
                    }
                    
                    return team
                  })
                })

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
          />
          ) : (
            <TabGroup className="gap-2 h-[75vh] w-full">
            <TabList className="flex justify-between mt-2 w-full items-center">
              <div className="flex">
                <Tab>Your Teams</Tab>
                <Tab>Available Teams</Tab>
                {isAdminRole(userRole || "") && <Tab>Default Team Settings</Tab>}
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
                        <div className="relative w-64">
                          <input
                            type="text"
                            placeholder="Search by Team Name..."
                            className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            value={filters.team_alias}
                            onChange={(e) => handleFilterChange('team_alias', e.target.value)}
                          />
                          <svg
                            className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                            />
                          </svg>
                        </div>

                        {/* Filter Button */}
                        <button
                          className={`px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 ${showFilters ? 'bg-gray-100' : ''}`}
                          onClick={() => setShowFilters(!showFilters)}
                        >
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                            />
                          </svg>
                          Filters
                          {(filters.team_id || filters.team_alias || filters.organization_id) && (
                            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                          )}
                        </button>

                        {/* Reset Filters Button */}
                        <button
                          className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                          onClick={handleFilterReset}
                        >
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                            />
                          </svg>
                          Reset Filters
                        </button>
                      </div>

                      {/* Additional Filters */}
                      {showFilters && (
                        <div className="flex flex-wrap items-center gap-3 mt-3">
                          {/* Team ID Search */}
                          <div className="relative w-64">
                            <input
                              type="text"
                              placeholder="Enter Team ID"
                              className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                              value={filters.team_id}
                              onChange={(e) => handleFilterChange('team_id', e.target.value)}
                            />
                            <svg
                              className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                              />
                            </svg>
                          </div>

                          {/* Organization Dropdown */}
                          <div className="w-64">
                            <Select
                              value={filters.organization_id || ""}
                              onValueChange={(value) => handleFilterChange('organization_id', value)}
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
                      </TableRow>
                    </TableHead>

                    <TableBody>
                      {teams && teams.length > 0
                        ? teams
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
                          {Number(team["spend"]).toFixed(4)}
                        </TableCell>
                        <TableCell
                          style={{
                            maxWidth: "4px",
                            whiteSpace: "pre-wrap",
                            overflow: "hidden",
                          }}
                        >
                          {team["max_budget"] !== null && team["max_budget"] !== undefined ? team["max_budget"] : "No limit"}
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
                                            icon={expandedAccordions[team.team_id] ? ChevronDownIcon : ChevronRightIcon}
                                            className="cursor-pointer"
                                            size="xs"
                                            onClick={() => {
                                              setExpandedAccordions(prev => ({
                                                ...prev,
                                                [team.team_id]: !prev[team.team_id]
                                              }));
                                            }}
                                          />
                                        </div>
                                      )}
                                      <div className="flex flex-wrap gap-1">
                                        {team.models.slice(0, 3).map((model: string, index: number) => (
                                          model === "all-proxy-models" ? (
                                            <Badge
                                              key={index}
                                              size={"xs"}
                                              color="red"
                                            >
                                              <Text>All Proxy Models</Text>
                                            </Badge>
                                          ) : (
                                            <Badge
                                              key={index}
                                              size={"xs"}
                                              color="blue"
                                            >
                                              <Text>
                                                {model.length > 30
                                                  ? `${getModelDisplayName(model).slice(0, 30)}...`
                                                  : getModelDisplayName(model)}
                                              </Text>
                                            </Badge>
                                          )
                                        ))}
                                        {team.models.length > 3 && !expandedAccordions[team.team_id] && (
                                          <Badge size={"xs"} color="gray" className="cursor-pointer">
                                            <Text>+{team.models.length - 3} {team.models.length - 3 === 1 ? 'more model' : 'more models'}</Text>
                                          </Badge>
                                        )}
                                        {expandedAccordions[team.team_id] && (
                                      <div className="flex flex-wrap gap-1">
                                        {team.models.slice(3).map((model: string, index: number) => (
                                          model === "all-proxy-models" ? (
                                            <Badge
                                              key={index + 3}
                                              size={"xs"}
                                              color="red"
                                            >
                                              <Text>All Proxy Models</Text>
                                            </Badge>
                                          ) : (
                                            <Badge
                                              key={index + 3}
                                              size={"xs"}
                                              color="blue"
                                            >
                                              <Text>
                                                {model.length > 30
                                                  ? `${getModelDisplayName(model).slice(0, 30)}...`
                                                  : getModelDisplayName(model)}
                                              </Text>
                                            </Badge>
                                          )
                                        ))}
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
                          {team.organization_id}
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
                            <Icon
                              icon={PencilAltIcon}
                              size="sm"
                              onClick={() => {
                                setSelectedTeamId(team.team_id);
                                setEditTeam(true);
                              }}
                            />
                            <Icon
                              onClick={() => handleDelete(team.team_id)}
                              icon={TrashIcon}
                              size="sm"
                            />
                            </>
                          ) : null}
                        </TableCell>
                      </TableRow>
                    ))
                  : null}
              </TableBody>
            </Table>
            {isDeleteModalOpen && (
              <div className="fixed z-10 inset-0 overflow-y-auto">
                <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                  <div
                    className="fixed inset-0 transition-opacity"
                    aria-hidden="true"
                  >
                    <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
                  </div>

                        {/* Modal Panel */}
                        <span
                          className="hidden sm:inline-block sm:align-middle sm:h-screen"
                          aria-hidden="true"
                        >
                          &#8203;
                        </span>

                        {/* Confirmation Modal Content */}
                        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                            <div className="sm:flex sm:items-start">
                              <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                                <h3 className="text-lg leading-6 font-medium text-gray-900">
                                  Delete Team
                                </h3>
                                <div className="mt-2">
                                  <p className="text-sm text-gray-500">
                                    Are you sure you want to delete this team ?
                                  </p>
                                </div>
                              </div>
                            </div>
                          </div>
                          <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                            <Button
                              onClick={confirmDelete}
                              color="red"
                              className="ml-2"
                            >
                              Delete
                            </Button>
                            <Button onClick={cancelDelete}>Cancel</Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </Card>
              </Col>
              {userRole == "Admin" || userRole == "Org Admin"? (
                <Col numColSpan={1}>
                <Modal
                  title="Create Team"
                  visible={isTeamModalVisible}
                  width={800}
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
                          { required: true, message: "Please input a team name" },
                        ]}
                      >
                        <TextInput placeholder="" />
                      </Form.Item>
                      <Form.Item
                        label={
                          <span>
                            Organization{' '}
                            <Tooltip title={
                              <span>
                                Organizations can have multiple teams. Learn more about{' '}
                                <a 
                                  href="https://docs.litellm.ai/docs/proxy/user_management_heirarchy"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  style={{ color: '#1890ff', textDecoration: 'underline' }}
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  user management hierarchy
                                </a>
                              </span>
                            }>
                              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                            </Tooltip>
                          </span>
                        }
                        name="organization_id"
                        initialValue={currentOrg ? currentOrg.organization_id : null}
                        className="mt-8"
                      >
                        <Select2
                          showSearch
                          allowClear
                          placeholder="Search or select an Organization"
                          onChange={(value) => {
                            form.setFieldValue('organization_id', value);
                            setCurrentOrgForCreateTeam(organizations?.find((org) => org.organization_id === value) || null);
                          }}
                          filterOption={(input, option) => {
                            if (!option) return false;
                            const optionValue = option.children?.toString() || '';
                            return optionValue.toLowerCase().includes(input.toLowerCase());
                          }}
                          optionFilterProp="children"
                        >
                          {organizations?.map((org) => (
                            <Select2.Option key={org.organization_id} value={org.organization_id}>
                              <span className="font-medium">{org.organization_alias}</span>{" "}
                              <span className="text-gray-500">({org.organization_id})</span>
                            </Select2.Option>
                          ))}
                        </Select2>
                      </Form.Item>
                      <Form.Item label={
                          <span>
                            Models{' '}
                            <Tooltip title="These are the models that your selected team has access to">
                              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                            </Tooltip>
                          </span>
                        } name="models">
                        <Select2
                          mode="multiple"
                          placeholder="Select models"
                          style={{ width: "100%" }}
                        >
                          <Select2.Option
                            key="all-proxy-models"
                            value="all-proxy-models"
                          >
                            All Proxy Models
                          </Select2.Option>
                          {modelsToPick.map((model) => (
                            <Select2.Option key={model} value={model}>
                              {getModelDisplayName(model)}
                            </Select2.Option>
                          ))}
                        </Select2>
                      </Form.Item>

                      <Form.Item label="Max Budget (USD)" name="max_budget">
                        <NumericalInput step={0.01} precision={2} width={200} />
                      </Form.Item>
                      <Form.Item
                        className="mt-8"
                        label="Reset Budget"
                        name="budget_duration"
                      >
                        <Select2 defaultValue={null} placeholder="n/a">
                          <Select2.Option value="24h">daily</Select2.Option>
                          <Select2.Option value="7d">weekly</Select2.Option>
                          <Select2.Option value="30d">monthly</Select2.Option>
                        </Select2>
                      </Form.Item>
                      <Form.Item
                        label="Tokens per minute Limit (TPM)"
                        name="tpm_limit"
                      >
                        <NumericalInput step={1} width={400} />
                      </Form.Item>
                      <Form.Item
                        label="Requests per minute Limit (RPM)"
                        name="rpm_limit"
                      >
                        <NumericalInput step={1} width={400} />
                      </Form.Item>

                      <Accordion className="mt-20 mb-8">
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
                          <Form.Item label="Metadata" name="metadata" help="Additional team metadata. Enter metadata as JSON object.">
                            <Input.TextArea rows={4} />
                          </Form.Item>
                          <Form.Item 
                            label={
                              <span>
                                Guardrails{' '}
                                <Tooltip title="Setup your first guardrail">
                                  <a 
                                    href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    <InfoCircleOutlined style={{ marginLeft: '4px' }} />
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
                              style={{ width: '100%' }}
                              placeholder="Select or enter guardrails"
                              options={guardrailsList.map(name => ({ value: name, label: name }))}
                            />
                          </Form.Item>
                          <Form.Item 
                            label={
                              <span>
                                Allowed Vector Stores{' '}
                                <Tooltip title="Select which vector stores this team can access by default. Leave empty for access to all vector stores">
                                  <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                                </Tooltip>
                              </span>
                            }
                            name="allowed_vector_store_ids"
                            className="mt-8"
                            help="Select vector stores this team can access. Leave empty for access to all vector stores"
                          >
                            <PremiumVectorStoreSelector
                              onChange={(values) => form.setFieldValue('allowed_vector_store_ids', values)}
                              value={form.getFieldValue('allowed_vector_store_ids')}
                              accessToken={accessToken || ''}
                              placeholder="Select vector stores (optional)"
                              premiumUser={premiumUser}
                            />
                          </Form.Item>
                        </AccordionBody>
                      </Accordion>
                    </>
                    <div style={{ textAlign: "right", marginTop: "10px" }}>
                      <Button2 htmlType="submit">Create Team</Button2>
                    </div>
                  </Form>
                </Modal>
                </Col>
              ) : null}
            </Grid>
            </TabPanel>
            <TabPanel>  
              <AvailableTeamsPanel
                accessToken={accessToken}
                userID={userID}
              />
            </TabPanel>
            {isAdminRole(userRole || "") && (
              <TabPanel>
                <TeamSSOSettings
                  accessToken={accessToken}
                  userID={userID || ""}
                  userRole={userRole || ""}
                />
              </TabPanel>
            )}
            </TabPanels>

            </TabGroup>)}
        </Col>
      </Grid>
    </div>
  );
};

export default Teams;
