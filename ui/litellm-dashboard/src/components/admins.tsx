/**
 * Allow proxy admin to add other people to view global spend
 * Use this to avoid sharing master key with others
 */
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
  Col,
  Text,
  Grid,
} from "@tremor/react";
import { CogIcon } from "@heroicons/react/outline";
interface AdminPanelProps {
  teams: any[] | null;
  searchParams: any;
  accessToken: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Object[] | null>>;
}
import { teamCreateCall, teamMemberAddCall, Member } from "./networking";

const AdminPanel: React.FC<AdminPanelProps> = ({
  teams,
  searchParams,
  accessToken,
  setTeams,
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

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null) {
        message.info("Making API Call");
        const response: any = await teamCreateCall(accessToken, formValues);
        if (teams !== null) {
          setTeams([...teams, response]);
        } else {
          setTeams([response]);
        }
        console.log(`response for team create call: ${response}`);
        setIsTeamModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };

  const handleMemberCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null && teams != null) {
        message.info("Making API Call");
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
    <div className="w-full m-2">
      <Title level={4}>Proxy Admins</Title>
      <Paragraph>Add other people to just view global spend.</Paragraph>
      <Grid numItems={1} className="gap-2 p-0 w-full">
        <Col numColSpan={1}>
          <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Member Name</TableHeaderCell>
                  <TableHeaderCell>Role</TableHeaderCell>
                  <TableHeaderCell>Action</TableHeaderCell>
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
                          <TableCell>
                            <Icon icon={CogIcon} size="sm" />
                          </TableCell>
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

export default AdminPanel;
