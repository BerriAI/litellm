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
import { teamInfoCall, teamMemberDeleteCall, teamMemberAddCall, teamMemberUpdateCall, Member, teamUpdateCall } from "@/components/networking";
import { Button, Form, Input, Select, message, InputNumber, Tooltip } from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import {
  Select as Select2,
} from "antd";
import { PencilAltIcon, PlusIcon, TrashIcon } from "@heroicons/react/outline";
import TeamMemberModal from "./edit_membership";
import UserSearchModal from "@/components/common_components/user_search_modal";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";


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
  userModels: string[];
  editTeam: boolean;
}

const TeamInfoView: React.FC<TeamInfoProps> = ({ 
  teamId, 
  onClose, 
  accessToken, 
  is_team_admin, 
  is_proxy_admin,
  userModels,
  editTeam
}) => {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  console.log("userModels in team info", userModels);

  const canManageMembers = is_team_admin || is_proxy_admin;
  const canEditTeam = is_team_admin || is_proxy_admin;

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

  const handleTeamUpdate = async (values: any) => {
    try {
      if (!accessToken) return;

      const updateData = {
        team_id: teamId,
        team_alias: values.team_alias,
        models: values.models,
        tpm_limit: values.tpm_limit,
        rpm_limit: values.rpm_limit,
        max_budget: values.max_budget,
        budget_duration: values.budget_duration,
        metadata: {
          ...teamData?.team_info?.metadata,
          guardrails: values.guardrails || []
        }
      };
      
      const response = await teamUpdateCall(accessToken, updateData);
      
      message.success("Team settings updated successfully");
      setIsEditing(false);
      fetchTeamInfo();
    } catch (error) {
      message.error("Failed to update team settings");
      console.error("Error updating team:", error);
    }
  };

  const renderSettingsPanel = () => {
    if (!teamData?.team_info) return null;
    const info = teamData.team_info;

    // Extract existing guardrails from team metadata
    let existingGuardrails: string[] = [];
    try {
      existingGuardrails = info.metadata?.guardrails || [];
    } catch (error) {
      console.error("Error extracting guardrails:", error);
    }

    if (!isEditing) {
      return (
        <Card>
          <div className="flex justify-between">
            <Title>Team Settings</Title>
            {canEditTeam && (
              <Button type="primary" onClick={() => setIsEditing(true)}>
                Edit Settings
              </Button>
            )}
          </div>
          <div className="mt-4 space-y-4">
            <div>
              <Text className="font-medium">Team Name</Text>
              <Text>{info.team_alias}</Text>
            </div>
            <div>
              <Text className="font-medium">Team ID</Text>
              <Text className="font-mono">{info.team_id}</Text>
            </div>
            <div>
              <Text className="font-medium">Created At</Text>
              <Text>{new Date(info.created_at).toLocaleString()}</Text>
            </div>
            <div>
              <Text className="font-medium">Models</Text>
              <div className="flex flex-wrap gap-2 mt-1">
                {info.models.map((model, index) => (
                  <Badge key={index} color="red">
                    {model}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <Text className="font-medium">Rate Limits</Text>
              <Text>TPM: {info.tpm_limit || 'Unlimited'}</Text>
              <Text>RPM: {info.rpm_limit || 'Unlimited'}</Text>
            </div>
            <div>
              <Text className="font-medium">Budget</Text>
              <Text>Max: ${info.max_budget || 'Unlimited'}</Text>
              <Text>Reset: {info.budget_duration || 'Never'}</Text>
            </div>
            <div>
              <Text className="font-medium">Status</Text>
              <Badge color={info.blocked ? 'red' : 'green'}>
                {info.blocked ? 'Blocked' : 'Active'}
              </Badge>
            </div>
          </div>
        </Card>
      );
    }

    return (
      <Card>
        <Title>Edit Team Settings</Title>
        <Form
          form={form}
          onFinish={handleTeamUpdate}
          initialValues={{
            ...info,
            guardrails: existingGuardrails
          }}
          layout="vertical"
          className="mt-4"
        >
          <Form.Item
            label="Team Name"
            name="team_alias"
            rules={[{ required: true, message: "Please input a team name" }]}
          >
            <Input />
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
            <InputNumber step={0.01} precision={2} style={{ width: 200 }} />
          </Form.Item>

          <Form.Item label="Reset Budget" name="budget_duration">
            <Select placeholder="n/a">
              <Select.Option value="24h">daily</Select.Option>
              <Select.Option value="7d">weekly</Select.Option>
              <Select.Option value="30d">monthly</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
            <InputNumber step={1} style={{ width: "100%" }} />
          </Form.Item>

          <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
            <InputNumber step={1} style={{ width: "100%" }} />
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
            help="Select existing guardrails or enter new ones"
          >
            <Select
              mode="tags"
              placeholder="Select or enter guardrails"
            />
          </Form.Item>

          <div className="flex justify-end gap-2 mt-6">
            <Button onClick={() => setIsEditing(false)}>
              Cancel
            </Button>
            <Button type="primary" htmlType="submit">
              Save Changes
            </Button>
          </div>
        </Form>
      </Card>
    );
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!teamData?.team_info) {
    return <div className="p-4">Team not found</div>;
  }

  const { team_info: info } = teamData;

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">← Back</Button>
          <Title>{info.team_alias}</Title>
          <Text className="text-gray-500 font-mono">{info.team_id}</Text>
        </div>
      </div>

      <TabGroup defaultIndex={editTeam ? 2 : 0}>
        <TabList className="mb-4">
          <Tab>Overview</Tab>
          <Tab>Members</Tab>
          <Tab>Settings</Tab>
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
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

          {/* Members Panel */}
          <TabPanel>
            <div className="space-y-4">
              <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>User ID</TableHeaderCell>
                      <TableHeaderCell>User Email</TableHeaderCell>
                      <TableHeaderCell>Role</TableHeaderCell>
                      <TableHeaderCell></TableHeaderCell>
                    </TableRow>
                  </TableHead>

                  <TableBody>
                    {teamData.team_info.members_with_roles.map((member: Member, index: number) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Text className="font-mono">{member.user_id}</Text>
                        </TableCell>
                        <TableCell>
                          <Text className="font-mono">{member.user_email}</Text>
                        </TableCell>
                        <TableCell>
                          <Text className="font-mono">{member.role}</Text>
                        </TableCell>
                        <TableCell>
                          {is_team_admin && (
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
                                onClick={() => handleMemberDelete(member)}
                                icon={TrashIcon}
                                size="sm"
                              />
                            </>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Card>
              <TremorButton onClick={() => setIsAddMemberModalVisible(true)}>
                Add Member
              </TremorButton>
            </div>
          </TabPanel>

          {/* Settings Panel */}
          <TabPanel>
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Team Settings</Title>
                {canEditTeam && (
                  <Button 
                    type="primary"
                    className="bg-blue-500"
                    onClick={() => setIsEditing(true)}
                  >
                    Edit Settings
                  </Button>
                )}
              </div>

              {isEditing ? (
                <Form
                  form={form}
                  onFinish={handleTeamUpdate}
                  initialValues={{
                    ...info,
                    team_alias: info.team_alias,
                    models: info.models,
                    tpm_limit: info.tpm_limit,
                    rpm_limit: info.rpm_limit,
                    max_budget: info.max_budget,
                    budget_duration: info.budget_duration,
                    guardrails: info.metadata?.guardrails || []
                  }}
                  layout="vertical"
                >
                  <Form.Item
                    label="Team Name"
                    name="team_alias"
                    rules={[{ required: true, message: "Please input a team name" }]}
                  >
                    <Input />
                  </Form.Item>
                  
                  <Form.Item label="Models" name="models">
                    <Select
                      mode="multiple"
                      placeholder="Select models"
                    >
                      <Select.Option key="all-proxy-models" value="all-proxy-models">
                        All Proxy Models
                      </Select.Option>
                      {userModels.map((model) => (
                        <Select.Option key={model} value={model}>
                          {getModelDisplayName(model)}
                        </Select.Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item label="Max Budget (USD)" name="max_budget">
                    <InputNumber step={0.01} precision={2} style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item label="Reset Budget" name="budget_duration">
                    <Select placeholder="n/a">
                      <Select.Option value="24h">daily</Select.Option>
                      <Select.Option value="7d">weekly</Select.Option>
                      <Select.Option value="30d">monthly</Select.Option>
                    </Select>
                  </Form.Item>

                  <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
                    <InputNumber step={1} style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
                    <InputNumber step={1} style={{ width: "100%" }} />
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
                    help="Select existing guardrails or enter new ones"
                  >
                    <Select
                      mode="tags"
                      placeholder="Select or enter guardrails"
                    />
                  </Form.Item>

                  <div className="flex justify-end gap-2">
                    <Button onClick={() => setIsEditing(false)}>
                      Cancel
                    </Button>
                    <Button type="primary" htmlType="submit" className="bg-blue-500">
                      Save Changes
                    </Button>
                  </div>
                </Form>
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Team Name</Text>
                    <div>{info.team_alias}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Team ID</Text>
                    <div className="font-mono">{info.team_id}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Created At</Text>
                    <div>{new Date(info.created_at).toLocaleString()}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Models</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {info.models.map((model, index) => (
                        <Badge key={index} color="red">
                          {model}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Rate Limits</Text>
                    <div>TPM: {info.tpm_limit || 'Unlimited'}</div>
                    <div>RPM: {info.rpm_limit || 'Unlimited'}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Budget</Text>
                    <div>Max: ${info.max_budget || 'Unlimited'}</div>
                    <div>Reset: {info.budget_duration || 'Never'}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Status</Text>
                    <Badge color={info.blocked ? 'red' : 'green'}>
                      {info.blocked ? 'Blocked' : 'Active'}
                    </Badge>
                  </div>
                </div>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <TeamMemberModal
        visible={isEditMemberModalVisible}
        onCancel={() => setIsEditMemberModalVisible(false)}
        onSubmit={handleMemberUpdate}
        initialData={selectedEditMember}
        mode="edit"
      />

      <UserSearchModal
        isVisible={isAddMemberModalVisible}
        onCancel={() => setIsAddMemberModalVisible(false)}
        onSubmit={handleMemberCreate}
        accessToken={accessToken}
      />
    </div>
  );
};

export default TeamInfoView;