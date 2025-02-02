import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Grid,
  Badge,
  Button as TremorButton,
  TableRow,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableBody,
  Table,
  Icon
} from "@tremor/react";
import { teamInfoCall, teamMemberDeleteCall, teamMemberAddCall, teamMemberUpdateCall, Member } from "@/components/networking";
import { Button, Modal, Form, Input, Select as AntSelect, message } from "antd";
import { PencilAltIcon, PlusIcon, TrashIcon } from "@heroicons/react/outline";
import TeamMemberModal from "./edit_membership";

interface TeamData {
  team_id: string;
  team_info: {
    team_alias: string;
    team_id: string;
    organization_id: string | null;
    admins: string[];
    members: string[];
    members_with_roles: Member[];
    metadata: Record<string, any>;
    tpm_limit: number | null;
    rpm_limit: number | null;
    max_budget: number | null;
    budget_duration: string | null;
    models: string[];
    blocked: boolean;
    spend: number;
    max_parallel_requests: number | null;
    budget_reset_at: string | null;
    model_id: string | null;
    litellm_model_table: string | null;
    created_at: string;
  };
  keys: any[];
  team_memberships: any[];
}

interface TeamInfoProps {
  teamId: string;
  onClose: () => void;
  accessToken: string | null;
  is_team_admin: boolean;
  is_proxy_admin: boolean;
}

const TeamInfoView: React.FC<TeamInfoProps> = ({ 
  teamId, 
  onClose, 
  accessToken, 
  is_team_admin, 
  is_proxy_admin 
}) => {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(null);

  const canManageMembers = is_team_admin || is_proxy_admin;

  const fetchTeamInfo = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await teamInfoCall(accessToken, teamId);
      setTeamData(response);
    } catch (error) {
      message.error("Failed to load team information");
      console.error("Error fetching team info:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTeamInfo();
  }, [teamId, accessToken]);


  const handleMemberCreate = async (values: any) => {
    try {
      if (accessToken == null) {
        return;
      }

      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      }
      const response = await teamMemberAddCall(accessToken, teamId, member);


      message.success("Team member added successfully");
      setIsAddMemberModalVisible(false);
      form.resetFields();
      fetchTeamInfo();
    } catch (error) {
      message.error("Failed to add team member");
      console.error("Error adding team member:", error);
    }
  };

  const handleMemberDelete = async (member: Member) => {
    try {
      if (accessToken == null) {
        return;
      }

      const response = await teamMemberDeleteCall(accessToken, teamId, member);

      message.success("Team member removed successfully");
      fetchTeamInfo();
    } catch (error) {
      message.error("Failed to remove team member");
      console.error("Error removing team member:", error);
    }
  };

  const handleMemberUpdate = async (values: any) => {
    try {
      if (accessToken == null) {
        return;
      }

      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      }

      const response = await teamMemberUpdateCall(accessToken, teamId, member);

      message.success("Team member updated successfully");
      setIsEditMemberModalVisible(false);
      fetchTeamInfo();
    } catch (error) {
      message.error("Failed to update team member");
      console.error("Error updating team member:", error);
    }
  }

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!teamData?.team_info) {
    return <div className="p-4">Team not found</div>;
  }

  const { team_info: info } = teamData;

  const renderMembersPanel = () => (
    <div className="space-y-4">
    <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>User ID</TableHeaderCell>
            <TableHeaderCell>User Email</TableHeaderCell>
            <TableHeaderCell>Role</TableHeaderCell>
          </TableRow>
        </TableHead>

        <TableBody>
          {teamData
            ? teamData.team_info.members_with_roles.map(
                (member: any, index: number) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Text className="font-mono">{member["user_id"]}</Text>
                    </TableCell>
                    <TableCell>
                      <Text className="font-mono">{member["user_email"]}</Text>
                    </TableCell>
                    <TableCell>
                      <Text className="font-mono">{member["role"]}</Text>
                    </TableCell>
                    <TableCell>
                    {is_team_admin ? (
                      <>
                      <Icon
                        icon={PencilAltIcon}
                        size="sm"
                        onClick={() => {
                          setSelectedEditMember(member);
                          setIsEditMemberModalVisible(true);
                        }}
                      />
                      <Icon
                        onClick={() => {handleMemberDelete(member)}}
                        icon={TrashIcon}
                        size="sm"
                      />
                      </>
                    ) : null}
                  </TableCell>
                  </TableRow>
                )
              )
            : null}
        </TableBody>
      </Table>
    </Card>
    <TremorButton onClick={() => setIsAddMemberModalVisible(true)}>Add Member</TremorButton>
    </div>
  );

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">← Back</Button>
          <Title>{info.team_alias}</Title>
          <Text className="text-gray-500 font-mono">{info.team_id}</Text>
        </div>
      </div>

      <TabGroup>
        <TabList className="mb-4">
          <Tab>Overview</Tab>
          <Tab>Members</Tab>
          <Tab>Settings</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Budget Status</Text>
                <div className="mt-2">
                  <Title>${info.spend.toFixed(6)}</Title>
                  <Text>of {info.max_budget === null ? "Unlimited" : `$${info.max_budget}`}</Text>
                  {info.budget_duration && (
                    <Text className="text-gray-500">Reset: {info.budget_duration}</Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Rate Limits</Text>
                <div className="mt-2">
                  <Text>TPM: {info.tpm_limit || 'Unlimited'}</Text>
                  <Text>RPM: {info.rpm_limit || 'Unlimited'}</Text>
                  {info.max_parallel_requests && (
                    <Text>Max Parallel Requests: {info.max_parallel_requests}</Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Models</Text>
                <div className="mt-2 flex flex-wrap gap-2">
                  {info.models.map((model, index) => (
                    <Badge key={index} color="red">
                      {model}
                    </Badge>
                  ))}
                </div>
              </Card>
            </Grid>
          </TabPanel>

          <TabPanel>
            {renderMembersPanel()}
          </TabPanel>

          <TabPanel>
            <Card>
              <Title>Team Settings</Title>
              <div className="mt-4 space-y-4">
                <div>
                  <Text className="font-medium">Team ID</Text>
                  <Text className="font-mono">{info.team_id}</Text>
                </div>
                <div>
                  <Text className="font-medium">Created At</Text>
                  <Text>{new Date(info.created_at).toLocaleString()}</Text>
                </div>
                <div>
                  <Text className="font-medium">Status</Text>
                  <Badge color={info.blocked ? 'red' : 'green'}>
                    {info.blocked ? 'Blocked' : 'Active'}
                  </Badge>
                </div>
              </div>
            </Card>
            <TeamMemberModal
            visible={isEditMemberModalVisible}
            onCancel={() => setIsEditMemberModalVisible(false)}
            onSubmit={handleMemberUpdate}
            initialData={selectedEditMember}
            mode="edit"
          />
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <Modal
        title="Add Team Member"
        open={isAddMemberModalVisible}
        onCancel={() => {
          setIsAddMemberModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={800}
      >
        <Form
          form={form}
          onFinish={handleMemberCreate}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
          initialValues={{
            role: "user",
          }}
        >
          <Form.Item label="Email" name="user_email" className="mb-4">
            <Input className="px-3 py-2 border rounded-md w-full" />
          </Form.Item>
          <div className="text-center mb-4">OR</div>
          <Form.Item label="User ID" name="user_id" className="mb-4">
            <Input className="px-3 py-2 border rounded-md w-full" />
          </Form.Item>
          <Form.Item label="Member Role" name="role" className="mb-4">
            <AntSelect defaultValue="user">
              <AntSelect.Option value="admin">admin</AntSelect.Option>
              <AntSelect.Option value="user">user</AntSelect.Option>
            </AntSelect>
          </Form.Item>
          <div className="text-right mt-4">
            <Button type="primary" htmlType="submit">
              Add Member
            </Button>
          </div>
        </Form>
      </Modal>
    </div>
  );
};

export default TeamInfoView;