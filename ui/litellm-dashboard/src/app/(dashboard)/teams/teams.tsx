import React, { useState, useEffect } from "react";
import { teamDeleteCall, Organization, fetchMCPAccessGroups } from "@/components/networking";
import { fetchTeams } from "@/components/common_components/fetch_teams";
import { Button as Button2, Modal, Form, Input, Select as Select2, Tooltip } from "antd";
import NumericalInput from "../../../components/shared/numerical_input";
import {
  fetchAvailableModelsForTeamOrKey,
  getModelDisplayName,
  unfurlWildcardModelsInList,
} from "@/components/key_team_helpers/fetch_available_models_team_key";
import { InfoCircleOutlined } from "@ant-design/icons";
import { getGuardrailsList } from "@/components/networking";
import TeamInfoView from "@/components/team/team_info";
import TeamSSOSettings from "@/components/TeamSSOSettings";
import { isAdminRole } from "@/utils/roles";
import {
  TextInput,
  Card,
  Button,
  Col,
  Text,
  Grid,
  Accordion,
  AccordionHeader,
  AccordionBody,
  TabPanel,
} from "@tremor/react";
import AvailableTeamsPanel from "@/components/team/available_teams";
import VectorStoreSelector from "../../../components/vector_store_management/VectorStoreSelector";
import PremiumLoggingSettings from "../../../components/common_components/PremiumLoggingSettings";
import type { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import { AlertTriangleIcon, XIcon } from "lucide-react";
import MCPServerSelector from "./mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "./mcp_server_management/MCPToolPermissions";
import ModelAliasManager from "./common_components/ModelAliasManager";
import NotificationsManager from "./molecules/notifications_manager";

import { teamCreateCall, Member, v2TeamListCall } from "@/components/networking";
import { updateExistingKeys } from "@/utils/dataUtils";
import TeamsHeaderTabs from "@/app/(dashboard)/teams/components/TeamsHeaderTabs";
import TeamsFilters from "@/app/(dashboard)/teams/components/TeamsFilters";
import useFetchTeams from "@/app/(dashboard)/teams/hooks/useFetchTeams";
import TeamsTable from "@/app/(dashboard)/teams/components/TeamsTable/TeamsTable";

interface TeamProps {
  teams: Team[] | null;
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

const Teams: React.FC<TeamProps> = ({
  teams,
  accessToken,
  setTeams,
  userID,
  userRole,
  organizations,
  premiumUser = false,
}) => {
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

  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();

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
  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [mcpAccessGroups, setMcpAccessGroups] = useState<string[]>([]);
  const [mcpAccessGroupsLoaded, setMcpAccessGroupsLoaded] = useState(false);
  const [deleteConfirmInput, setDeleteConfirmInput] = useState("");
  const [modelAliases, setModelAliases] = useState<{ [key: string]: string }>({});
  const { lastRefreshed, onRefreshClick: handleRefreshClick } = useFetchTeams({ currentOrg, setTeams });

  useEffect(() => {
    console.log(`currentOrgForCreateTeam: ${currentOrgForCreateTeam}`);
    const models = getOrganizationModels(currentOrgForCreateTeam, userModels);
    console.log(`models: ${models}`);
    setModelsToPick(models);
    form.setFieldValue("models", []);
  }, [currentOrgForCreateTeam, userModels]);

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

    fetchGuardrails();
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
      fetchTeams(accessToken, userID, userRole, currentOrg, setTeams);
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

        // Add model_aliases if any are defined
        if (Object.keys(modelAliases).length > 0) {
          formValues.model_aliases = modelAliases;
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
          {(userRole == "Admin" || userRole == "Org Admin") && (
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
            />
          ) : (
            <TeamsHeaderTabs lastRefreshed={lastRefreshed} onRefresh={handleRefreshClick} userRole={userRole}>
              <TabPanel>
                <Text>
                  Click on &ldquo;Team ID&rdquo; to view team details <b>and</b> manage team members.
                </Text>
                <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
                  <Col numColSpan={1}>
                    <Card className="w-full mx-auto flex-auto overflow-hidden overflow-y-auto max-h-[50vh]">
                      <div className="border-b px-6 py-4">
                        <div className="flex flex-col space-y-4">
                          <TeamsFilters
                            filters={filters}
                            organizations={organizations}
                            showFilters={showFilters}
                            onToggleFilters={setShowFilters}
                            onChange={handleFilterChange}
                            onReset={handleFilterReset}
                          />
                        </div>
                      </div>
                      <TeamsTable
                        teams={teams}
                        currentOrg={currentOrg}
                        perTeamInfo={perTeamInfo}
                        userRole={userRole}
                        setSelectedTeamId={setSelectedTeamId}
                        setEditTeam={setEditTeam}
                        onDeleteTeam={handleDelete}
                      />
                      {isDeleteModalOpen &&
                        (() => {
                          const team = teams?.find((t) => t.team_id === teamToDelete);
                          const teamName = team?.team_alias || "";
                          const keyCount = team?.keys?.length || 0;
                          const isValid = deleteConfirmInput === teamName;
                          return (
                            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                              <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl min-h-[380px] py-6 overflow-hidden transform transition-all flex flex-col justify-between">
                                <div>
                                  <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                                    <h3 className="text-lg font-semibold text-gray-900">Delete Team</h3>
                                    <button
                                      onClick={() => {
                                        cancelDelete();
                                        setDeleteConfirmInput("");
                                      }}
                                      className="text-gray-400 hover:text-gray-500 focus:outline-none"
                                    >
                                      <XIcon size={20} />
                                    </button>
                                  </div>
                                  <div className="px-6 py-4">
                                    {keyCount > 0 && (
                                      <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-100 rounded-md mb-5">
                                        <div className="text-red-500 mt-0.5">
                                          <AlertTriangleIcon size={20} />
                                        </div>
                                        <div>
                                          <p className="text-base font-medium text-red-600">
                                            Warning: This team has {keyCount} associated key{keyCount > 1 ? "s" : ""}.
                                          </p>
                                          <p className="text-base text-red-600 mt-2">
                                            Deleting the team will also delete all associated keys. This action is
                                            irreversible.
                                          </p>
                                        </div>
                                      </div>
                                    )}
                                    <p className="text-base text-gray-600 mb-5">
                                      Are you sure you want to force delete this team and all its keys?
                                    </p>
                                    <div className="mb-5">
                                      <label className="block text-base font-medium text-gray-700 mb-2">
                                        {`Type `}
                                        <span className="underline">{teamName}</span>
                                        {` to confirm deletion:`}
                                      </label>
                                      <input
                                        type="text"
                                        value={deleteConfirmInput}
                                        onChange={(e) => setDeleteConfirmInput(e.target.value)}
                                        placeholder="Enter team name exactly"
                                        className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base"
                                        autoFocus
                                      />
                                    </div>
                                  </div>
                                </div>
                                <div className="px-6 py-4 bg-gray-50 flex justify-end gap-4">
                                  <button
                                    onClick={() => {
                                      cancelDelete();
                                      setDeleteConfirmInput("");
                                    }}
                                    className="px-5 py-3 bg-white border border-gray-300 rounded-md text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                                  >
                                    Cancel
                                  </button>
                                  <button
                                    onClick={confirmDelete}
                                    disabled={!isValid}
                                    className={`px-5 py-3 rounded-md text-base font-medium text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 ${isValid ? "bg-red-600 hover:bg-red-700" : "bg-red-300 cursor-not-allowed"}`}
                                  >
                                    Force Delete
                                  </button>
                                </div>
                              </div>
                            </div>
                          );
                        })()}
                    </Card>
                  </Col>
                </Grid>
              </TabPanel>
              <TabPanel>
                <AvailableTeamsPanel accessToken={accessToken} userID={userID} />
              </TabPanel>
              {isAdminRole(userRole || "") && (
                <TabPanel>
                  <TeamSSOSettings accessToken={accessToken} userID={userID || ""} userRole={userRole || ""} />
                </TabPanel>
              )}
            </TeamsHeaderTabs>
          )}
          {(userRole == "Admin" || userRole == "Org Admin") && (
            <Modal
              title="Create Team"
              visible={isTeamModalVisible}
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
                  >
                    <Select2
                      showSearch
                      allowClear
                      placeholder="Search or select an Organization"
                      onChange={(value) => {
                        form.setFieldValue("organization_id", value);
                        setCurrentOrgForCreateTeam(organizations?.find((org) => org.organization_id === value) || null);
                      }}
                      filterOption={(input, option) => {
                        if (!option) return false;
                        const optionValue = option.children?.toString() || "";
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
                  <Form.Item
                    label={
                      <span>
                        Models{" "}
                        <Tooltip title="These are the models that your selected team has access to">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="models"
                  >
                    <Select2 mode="multiple" placeholder="Select models" style={{ width: "100%" }}>
                      <Select2.Option key="all-proxy-models" value="all-proxy-models">
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
