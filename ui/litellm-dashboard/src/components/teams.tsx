import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Typography } from "antd";
import { teamDeleteCall, teamUpdateCall, teamInfoCall } from "./networking";
import {
  InformationCircleIcon,
  PencilAltIcon,
  PencilIcon,
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
import { Select, SelectItem } from "@tremor/react";
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
} from "@tremor/react";
import { CogIcon } from "@heroicons/react/outline";
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

  useEffect(() => {
    console.log(`inside useeffect - ${teams}`)
    if (teams === null && accessToken) {
      // Call your function here
      const fetchData = async () => {
        const givenTeams = await teamListCall(accessToken)
        console.log(`givenTeams: ${givenTeams}`)

        setTeams(givenTeams)
      }
      fetchData()
    }
  }, [teams]);

  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();
  const { Title, Paragraph } = Typography;
  const [value, setValue] = useState("");
  const [editModalVisible, setEditModalVisible] = useState(false);

  const [selectedTeam, setSelectedTeam] = useState<null | any>(
    teams ? teams[0] : null
  );
  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [userModels, setUserModels] = useState([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [teamToDelete, setTeamToDelete] = useState<string | null>(null);

  // store team info as {"team_id": team_info_object}
  const [perTeamInfo, setPerTeamInfo] = useState<Record<string, any>>({});

  const EditTeamModal: React.FC<EditTeamModalProps> = ({
    visible,
    onCancel,
    team,
    onSubmit,
  }) => {
    const [form] = Form.useForm();

    const handleOk = () => {
      form
        .validateFields()
        .then((values) => {
          const updatedValues = { ...values, team_id: team.team_id };
          onSubmit(updatedValues);
          form.resetFields();
        })
        .catch((error) => {
          console.error("Validation failed:", error);
        });
    };

    return (
      <Modal
        title="Edit Team"
        visible={visible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={onCancel}
      >
        <Form
          form={form}
          onFinish={handleEditSubmit}
          initialValues={team} // Pass initial values here
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            <Form.Item
              label="Team Name"
              name="team_alias"
              rules={[{ required: true, message: "Please input a team name" }]}
            >
              <TextInput />
            </Form.Item>
            <Form.Item label="Models" name="models">
              <Select2
                mode="multiple"
                placeholder="Select models"
                style={{ width: "100%" }}
              >
                <Select2.Option key="all-proxy-models" value="all-proxy-models">
                  {"All Proxy Models"}
                </Select2.Option>
                {userModels &&
                  userModels.map((model) => (
                    <Select2.Option key={model} value={model}>
                      {model}
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
            <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
              <InputNumber step={1} width={400} />
            </Form.Item>
            <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
              <InputNumber step={1} width={400} />
            </Form.Item>
            <Form.Item
              label="Requests per minute Limit (RPM)"
              name="team_id"
              hidden={true}
            ></Form.Item>
          </>
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Edit Team</Button2>
          </div>
        </Form>
      </Modal>
    );
  };

  const handleEditClick = (team: any) => {
    setSelectedTeam(team);
    setEditModalVisible(true);
  };

  const handleEditCancel = () => {
    setEditModalVisible(false);
    setSelectedTeam(null);
  };

  const handleEditSubmit = async (formValues: Record<string, any>) => {
    // Call API to update team with teamId and values
    const teamId = formValues.team_id; // get team_id

    console.log("handleEditSubmit:", formValues);
    if (accessToken == null) {
      return;
    }

    let newTeamValues = await teamUpdateCall(accessToken, formValues);

    // Update the teams state with the updated team data
    if (teams) {
      const updatedTeams = teams.map((team) =>
        team.team_id === teamId ? newTeamValues.data : team
      );
      setTeams(updatedTeams);
    }
    message.success("Team updated successfully");

    setEditModalVisible(false);
    setSelectedTeam(null);
  };

  const handleOk = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
  };

  const handleMemberOk = () => {
    setIsAddMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleCancel = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
  };

  const handleMemberCancel = () => {
    setIsAddMemberModalVisible(false);
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
        if (userID === null || userRole === null) {
          return;
        }

        if (accessToken !== null) {
          const model_available = await modelAvailableCall(
            accessToken,
            userID,
            userRole
          );
          let available_model_names = model_available["data"].map(
            (element: { id: string }) => element.id
          );
          console.log("available_model_names:", available_model_names);
          setUserModels(available_model_names);
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
        const teamList = await teamListCall(accessToken)
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
      if (accessToken != null) {
        const newTeamAlias = formValues?.team_alias;
        const existingTeamAliases = teams?.map((t) => t.team_alias) ?? [];

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

  const handleMemberCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null && teams != null) {
        message.info("Adding Member");
        const user_role: Member = {
          role: "user",
          user_email: formValues.user_email,
          user_id: formValues.user_id,
        };
        const response: any = await teamMemberAddCall(
          accessToken,
          selectedTeam["team_id"],
          user_role
        );
        console.log(`response for team create call: ${response["data"]}`);
        // Checking if the team exists in the list and updating or adding accordingly
        const foundIndex = teams.findIndex((team) => {
          console.log(
            `team.team_id=${team.team_id}; response.data.team_id=${response.data.team_id}`
          );
          return team.team_id === response.data.team_id;
        });
        console.log(`foundIndex: ${foundIndex}`);
        if (foundIndex !== -1) {
          // If the team is found, update it
          const updatedTeams = [...teams]; // Copy the current state
          updatedTeams[foundIndex] = response.data; // Update the specific team
          setTeams(updatedTeams); // Set the new state
          setSelectedTeam(response.data);
        }
        setIsAddMemberModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the team:", error);
    }
  };
  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 h-[75vh] w-full mt-2">
        <Col numColSpan={1}>
          <Title level={4}>All Teams</Title>
          <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Team Name</TableHeaderCell>
                  <TableHeaderCell>Team ID</TableHeaderCell>
                  <TableHeaderCell>Spend (USD)</TableHeaderCell>
                  <TableHeaderCell>Budget (USD)</TableHeaderCell>
                  <TableHeaderCell>Models</TableHeaderCell>
                  <TableHeaderCell>TPM / RPM Limits</TableHeaderCell>
                  <TableHeaderCell>Info</TableHeaderCell>
                </TableRow>
              </TableHead>

              <TableBody>
                {teams && teams.length > 0
                  ? teams.map((team: any) => (
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
                        <TableCell
                          style={{
                            maxWidth: "4px",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            fontSize: "0.75em", // or any smaller size as needed
                          }}
                        >
                          <Tooltip title={team.team_id}>
                          {team.team_id}
                          </Tooltip>
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
                                            ? `${model.slice(0, 30)}...`
                                            : model}
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
                          <Icon
                            icon={PencilAltIcon}
                            size="sm"
                            onClick={() => handleEditClick(team)}
                          />
                          <Icon
                            onClick={() => handleDelete(team.team_id)}
                            icon={TrashIcon}
                            size="sm"
                          />
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
                        {model}
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
              </>
              <div style={{ textAlign: "right", marginTop: "10px" }}>
                <Button2 htmlType="submit">Create Team</Button2>
              </div>
            </Form>
          </Modal>
        </Col>
        <Col numColSpan={1}>
          <Title level={4}>Team Members</Title>
          <Paragraph>
            If you belong to multiple teams, this setting controls which teams
            members you see.
          </Paragraph>
          {teams && teams.length > 0 ? (
            <Select defaultValue="0">
              {teams.map((team: any, index) => (
                <SelectItem
                  key={index}
                  value={String(index)}
                  onClick={() => {
                    setSelectedTeam(team);
                  }}
                >
                  {team["team_alias"]}
                </SelectItem>
              ))}
            </Select>
          ) : (
            <Paragraph>
              No team created. <b>Defaulting to personal account.</b>
            </Paragraph>
          )}
        </Col>
        <Col numColSpan={1}>
          <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Member Name</TableHeaderCell>
                  <TableHeaderCell>Role</TableHeaderCell>
                </TableRow>
              </TableHead>

              <TableBody>
                {selectedTeam
                  ? selectedTeam["members_with_roles"].map(
                      (member: any, index: number) => (
                        <TableRow key={index}>
                          <TableCell>
                            {member["user_email"]
                              ? member["user_email"]
                              : member["user_id"]
                                ? member["user_id"]
                                : null}
                          </TableCell>
                          <TableCell>{member["role"]}</TableCell>
                        </TableRow>
                      )
                    )
                  : null}
              </TableBody>
            </Table>
          </Card>
          {selectedTeam && (
            <EditTeamModal
              visible={editModalVisible}
              onCancel={handleEditCancel}
              team={selectedTeam}
              onSubmit={handleEditSubmit}
            />
          )}
        </Col>
        <Col numColSpan={1}>
          <Button
            className="mx-auto mb-5"
            onClick={() => setIsAddMemberModalVisible(true)}
          >
            + Add member
          </Button>
          <Modal
            title="Add member"
            visible={isAddMemberModalVisible}
            width={800}
            footer={null}
            onOk={handleMemberOk}
            onCancel={handleMemberCancel}
          >
            <Form
              form={form}
              onFinish={handleMemberCreate}
              labelCol={{ span: 8 }}
              wrapperCol={{ span: 16 }}
              labelAlign="left"
            >
              <>
                <Form.Item label="Email" name="user_email" className="mb-4">
                  <Input
                    name="user_email"
                    className="px-3 py-2 border rounded-md w-full"
                  />
                </Form.Item>
                <div className="text-center mb-4">OR</div>
                <Form.Item label="User ID" name="user_id" className="mb-4">
                  <Input
                    name="user_id"
                    className="px-3 py-2 border rounded-md w-full"
                  />
                </Form.Item>
              </>
              <div style={{ textAlign: "right", marginTop: "10px" }}>
                <Button2 htmlType="submit">Add member</Button2>
              </div>
            </Form>
          </Modal>
        </Col>
      </Grid>
    </div>
  );
};

export default Team;
