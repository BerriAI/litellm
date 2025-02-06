import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Typography } from "antd";
import { teamDeleteCall, teamUpdateCall, teamInfoCall } from "./networking";
import TeamMemberModal from "@/components/team/edit_membership";
import {
  InformationCircleIcon,
  PencilAltIcon,
  PencilIcon,
  RefreshIcon,
  StatusOnlineIcon,
  TrashIcon,
} from "@heroicons/react/outline";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  InputNumber,
  message,
  Tooltip
} from "antd";
import { fetchAvailableModelsForTeamOrKey, getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { Select, SelectItem } from "@tremor/react";
import { InfoCircleOutlined } from '@ant-design/icons';
import { getGuardrailsList } from "./networking";
import TeamInfoView from "@/components/team/team_info";
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
const isLocal = process.env.NODE_ENV === "development";
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;
if (isLocal != true) {
  console.log = function() {};
}
interface TeamProps {
  teams: any[] | null;
  searchParams: any;
  accessToken: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Object[] | null>>;
  userID: string | null;
  userRole: string | null;
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
  teamListCall
} from "./networking";


const Team: React.FC<TeamProps> = ({
  teams,
  searchParams,
  accessToken,
  setTeams,
  userID,
  userRole,
}) => {
  const [lastRefreshed, setLastRefreshed] = useState("");

  const fetchTeams = async (accessToken: string, userID: string | null, userRole: string | null) => {
    let givenTeams;
    if (userRole != "Admin" && userRole != "Admin Viewer") {
      givenTeams = await teamListCall(accessToken, userID)
    } else {
      givenTeams = await teamListCall(accessToken)
    }
    
    console.log(`givenTeams: ${givenTeams}`)

    setTeams(givenTeams)
  }
  useEffect(() => {
    console.log(`inside useeffect - ${teams}`)
    if (teams === null && accessToken) {
      // Call your function here
      fetchTeams(accessToken, userID, userRole)
    }
  }, [teams]);
  
  useEffect(() => {
    console.log(`inside useeffect - ${lastRefreshed}`)
    if (accessToken) {
      // Call your function here
      fetchTeams(accessToken, userID, userRole)
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
  


  const [perTeamInfo, setPerTeamInfo] = useState<Record<string, any>>({});

  // Add this state near the other useState declarations
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);

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
      const filteredData = teams.filter(
        (item) => item.team_id !== teamToDelete
      );
      setTeams(filteredData);
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

    const fetchTeamInfo = async () => {
      try {
        if (userID === null || userRole === null || accessToken === null) {
          return;
        }

        if (teams === null) {
          return;
        }

        let _team_id_to_info: Record<string, any> = {};
        let teamList;
        if (userRole != "Admin" && userRole != "Admin Viewer") {
          teamList = await teamListCall(accessToken, userID)
        } else {
          teamList = await teamListCall(accessToken)
        }
        
        for (let i = 0; i < teamList.length; i++) {
          let team = teamList[i];
          let _team_id = team.team_id;
      
          // Use the team info directly from the teamList
          if (team !== null) {
              _team_id_to_info = { ..._team_id_to_info, [_team_id]: team };
          }
        }
        setPerTeamInfo(_team_id_to_info);
      } catch (error) {
        console.error("Error fetching team info:", error);
      }
    };

    fetchUserModels();
    fetchTeamInfo();
  }, [accessToken, userID, userRole, teams]);

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      console.log(`formValues: ${JSON.stringify(formValues)}`);
      if (accessToken != null) {
        const newTeamAlias = formValues?.team_alias;
        const existingTeamAliases = teams?.map((t) => t.team_alias) ?? [];
        let organizationId = formValues?.organization_id;
        if (organizationId === "" || typeof organizationId !== 'string') {
          formValues.organization_id = null;
        } else {
          formValues.organization_id = organizationId.trim();
        }
        
        // Create metadata object with guardrails if they exist
        formValues.metadata = {
          ...(formValues.guardrails ? { guardrails: formValues.guardrails } : {})
        };
        
        // Remove guardrails from top level since it's now in metadata
        delete formValues.guardrails;

        if (existingTeamAliases.includes(newTeamAlias)) {
          throw new Error(
            `Team alias ${newTeamAlias} already exists, please pick another alias`
          );
        }

        message.info("Creating Team");
        const response: any = await teamCreateCall(accessToken, formValues);
        if (teams !== null) {
          setTeams([...teams, response]);
        } else {
          setTeams([response]);
        }
        console.log(`response for team create call: ${response}`);
        message.success("Team created");
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


  return (
    <div className="w-full mx-4">
      {selectedTeamId ? (
        <TeamInfoView 
        teamId={selectedTeamId} 
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
      <TabGroup className="gap-2 p-8 h-[75vh] w-full mt-2">
      <TabList className="flex justify-between mt-2 w-full items-center">
        <div className="flex">
          <Tab>Your Teams</Tab>
          <Tab>Available Teams</Tab>
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
          <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Team Name</TableHeaderCell>
                  <TableHeaderCell>Team ID</TableHeaderCell>
                  <TableHeaderCell>Created</TableHeaderCell>
                  <TableHeaderCell>Spend (USD)</TableHeaderCell>
                  <TableHeaderCell>Budget (USD)</TableHeaderCell>
                  <TableHeaderCell>Models</TableHeaderCell>
                  <TableHeaderCell>TPM / RPM Limits</TableHeaderCell>
                  <TableHeaderCell>Info</TableHeaderCell>
                </TableRow>
              </TableHead>

              <TableBody>
                {teams && teams.length > 0
                  ? teams
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
                        <TableRow>
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
                      </TableRow>

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
                          {team["spend"]}
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
                        >
                          {Array.isArray(team.models) ? (
                            <div
                              style={{
                                display: "flex",
                                flexDirection: "column",
                              }}
                            >
                              {team.models.length === 0 ? (
                                <Badge size={"xs"} className="mb-1" color="red">
                                  <Text>All Proxy Models</Text>
                                </Badge>
                              ) : (
                                team.models.map(
                                  (model: string, index: number) =>
                                    model === "all-proxy-models" ? (
                                      <Badge
                                        key={index}
                                        size={"xs"}
                                        className="mb-1"
                                        color="red"
                                      >
                                        <Text>All Proxy Models</Text>
                                      </Badge>
                                    ) : (
                                      <Badge
                                        key={index}
                                        size={"xs"}
                                        className="mb-1"
                                        color="blue"
                                      >
                                        <Text>
                                          {model.length > 30
                                            ? `${getModelDisplayName(model).slice(0, 30)}...`
                                            : getModelDisplayName(model)}
                                        </Text>
                                      </Badge>
                                    )
                                )
                              )}
                            </div>
                          ) : null}
                        </TableCell>

                        <TableCell
                          style={{
                            maxWidth: "4px",
                            whiteSpace: "pre-wrap",
                            overflow: "hidden",
                          }}
                        >
                          <Text>
                            TPM: {team.tpm_limit ? team.tpm_limit : "Unlimited"}{" "}
                            <br></br>RPM:{" "}
                            {team.rpm_limit ? team.rpm_limit : "Unlimited"}
                          </Text>
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
                              perTeamInfo[team.team_id].members_with_roles &&
                              perTeamInfo[team.team_id].members_with_roles.length}{" "}
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
        {userRole == "Admin"? (
          <Col numColSpan={1}>
            <Button
              className="mx-auto"
              onClick={() => setIsTeamModalVisible(true)}
            >
            + Create New Team
          </Button>
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
                <Form.Item label="Models" name="models">
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
                    {userModels.map((model) => (
                      <Select2.Option key={model} value={model}>
                        {getModelDisplayName(model)}
                      </Select2.Option>
                    ))}
                  </Select2>
                </Form.Item>

                <Form.Item label="Max Budget (USD)" name="max_budget">
                  <InputNumber step={0.01} precision={2} width={200} />
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
                  <InputNumber step={1} width={400} />
                </Form.Item>
                <Form.Item
                  label="Requests per minute Limit (RPM)"
                  name="rpm_limit"
                >
                  <InputNumber step={1} width={400} />
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
                    <Form.Item
                      label="Organization ID"
                      name="organization_id"
                      help="Assign team to an organization. Found in the 'Organization' tab."
                    >
                      <TextInput 
                        placeholder="" 
                        onChange={(e) => {
                          e.target.value = e.target.value.trim();
                        }} 
                      />
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
      </TabPanels>

      </TabGroup>)}
    </div>
  );
};

export default Team;
