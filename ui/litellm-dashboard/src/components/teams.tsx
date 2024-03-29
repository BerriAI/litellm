import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Typography } from "antd";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  InputNumber,
  message,
} from "antd";
import { Select, SelectItem } from "@tremor/react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Card,
  Icon,
  Button,
  Badge,
  Col,
  Text,
  Grid,
} from "@tremor/react";
import { CogIcon } from "@heroicons/react/outline";
interface TeamProps {
  teams: any[] | null;
  searchParams: any;
  accessToken: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Object[] | null>>;
  userID: string | null;
  userRole: string | null;
}
import { teamCreateCall, teamMemberAddCall, Member, modelAvailableCall } from "./networking";

const Team: React.FC<TeamProps> = ({
  teams,
  searchParams,
  accessToken,
  setTeams,
  userID,
  userRole,
}) => {
  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();
  const { Title, Paragraph } = Typography;
  const [value, setValue] = useState("");

  const [selectedTeam, setSelectedTeam] = useState<null | any>(
    teams ? teams[0] : null
  );
  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [userModels, setUserModels] = useState([]);

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

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null) {
          return;
        }

        if (accessToken !== null) {
          const model_available = await modelAvailableCall(accessToken, userID, userRole);
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
  
    fetchUserModels();
  }, [accessToken, userID, userRole]);

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null) {
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
      console.error("Error creating the key:", error);
      message.error("Error creating the team: " + error);
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
      console.error("Error creating the key:", error);
    }
  };
  console.log(`received teams ${teams}`);
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
                  <TableHeaderCell>Spend (USD)</TableHeaderCell>
                  <TableHeaderCell>Budget (USD)</TableHeaderCell>
                  <TableHeaderCell>Models</TableHeaderCell>
                  <TableHeaderCell>TPM / RPM Limits</TableHeaderCell>
                </TableRow>
              </TableHead>

              <TableBody>
                {teams && teams.length > 0
                  ? teams.map((team: any) => (
                      <TableRow key={team.team_id}>
                        <TableCell style={{ maxWidth: "4px", whiteSpace: "pre-wrap", overflow: "hidden"  }}>{team["team_alias"]}</TableCell>
                        <TableCell style={{ maxWidth: "4px", whiteSpace: "pre-wrap", overflow: "hidden"  }}>{team["spend"]}</TableCell>
                        <TableCell style={{ maxWidth: "4px", whiteSpace: "pre-wrap", overflow: "hidden"  }}>
                          {team["max_budget"] ? team["max_budget"] : "No limit"}
                        </TableCell>
                          <TableCell style={{ maxWidth: "8-x", whiteSpace: "pre-wrap", overflow: "hidden" }}>
                            {Array.isArray(team.models) ? (
                              <div style={{ display: "flex", flexDirection: "column" }}>
                                {team.models.map((model: string, index: number) => (
                                  <Badge key={index} size={"xs"} className="mb-1" color="blue">
                                    {model.length > 30 ? `${model.slice(0, 30)}...` : model}
                                  </Badge>
                                ))}
                              </div>
                            ) : null}
                        </TableCell>
                        <TableCell style={{ maxWidth: "4px", whiteSpace: "pre-wrap", overflow: "hidden"  }}>
                          <Text>
                            TPM:{" "}
                            {team.tpm_limit ? team.tpm_limit : "Unlimited"}{" "}
                            <br></br>RPM:{" "}
                            {team.rpm_limit ? team.rpm_limit : "Unlimited"}
                          </Text>
                        </TableCell>
                        {/* <TableCell>
                          <Icon icon={CogIcon} size="sm" />
                        </TableCell> */}
                      </TableRow>
                    ))
                  : null}
              </TableBody>
            </Table>
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
                  rules={[{ required: true, message: 'Please input a team name' }]}
                >
                  <Input />
                </Form.Item>
                <Form.Item label="Models" name="models">
                  <Select2
                    mode="multiple"
                    placeholder="Select models"
                    style={{ width: "100%" }}
                  >
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
                  {/* <TableHeaderCell>Action</TableHeaderCell> */}
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
                          {/* <TableCell>
                            <Icon icon={CogIcon} size="sm" />
                          </TableCell> */}
                        </TableRow>
                      )
                    )
                  : null}
              </TableBody>
            </Table>
          </Card>
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
