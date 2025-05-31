import React, { useState, useEffect } from "react";
import NumericalInput from "../shared/numerical_input";
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
import TeamMembersComponent from "./team_member_view";
import MemberPermissions from "./member_permissions";
import { teamInfoCall, teamMemberDeleteCall, teamMemberAddCall, teamMemberUpdateCall, Member, teamUpdateCall } from "@/components/networking";
import { Button, Form, Input, Select, message, Tooltip } from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import {
  Select as Select2,
} from "antd";
import { PencilAltIcon, PlusIcon, TrashIcon } from "@heroicons/react/outline";
import MemberModal from "./edit_membership";
import UserSearchModal from "@/components/common_components/user_search_modal";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { isAdminRole } from "@/utils/roles";
import ObjectPermissionsView from "../object_permissions_view";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import PremiumVectorStoreSelector from "../common_components/PremiumVectorStoreSelector";

export interface TeamData {
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
    litellm_model_table: {
      model_aliases: Record<string, string>;
    } | null;
    created_at: string;
    object_permission?: {
      object_permission_id: string;
      mcp_servers: string[];
      vector_stores: string[];
    };
  };
  keys: any[];
  team_memberships: any[];
}

export interface TeamInfoProps {
  teamId: string;
  onUpdate: (data: any) => void;
  onClose: () => void;
  accessToken: string | null;
  is_team_admin: boolean;
  is_proxy_admin: boolean;
  userModels: string[];
  editTeam: boolean;
  premiumUser?: boolean;
}

const TeamInfoView: React.FC<TeamInfoProps> = ({ 
  teamId, 
  onClose, 
  accessToken, 
  is_team_admin, 
  is_proxy_admin,
  userModels,
  editTeam,
  premiumUser = false
}) => {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  console.log("userModels in team info", userModels);

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
      if (accessToken == null) return;
  
      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };
  
      await teamMemberAddCall(accessToken, teamId, member);
  
      message.success("Team member added successfully");
      setIsAddMemberModalVisible(false);
      form.resetFields();
      fetchTeamInfo();
    } catch (error: any) {
      let errMsg = "Failed to add team member";
  
      if (error?.raw?.detail?.error?.includes("Assigning team admins is a premium feature")) {
        errMsg = "Assigning admins is an enterprise-only feature. Please upgrade your LiteLLM plan to enable this.";
      } else if (error?.message) {
        errMsg = error.message;
      }
  
      message.error(errMsg);
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
      message.destroy(); // Remove all existing toasts

      await teamMemberUpdateCall(accessToken, teamId, member);

      message.success("Team member updated successfully");
      setIsEditMemberModalVisible(false);
      fetchTeamInfo();
    } catch (error: any) {
      let errMsg = "Failed to update team member";
      if (error?.raw?.detail?.includes("Assigning team admins is a premium feature")) {
        errMsg = "Assigning admins is an enterprise-only feature. Please upgrade your LiteLLM plan to enable this.";
      } else if (error?.message) {
        errMsg = error.message;
      }
      setIsEditMemberModalVisible(false);

      message.destroy(); // Remove all existing toasts

      message.error(errMsg);
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

      let parsedMetadata = {};
      try {
        parsedMetadata = values.metadata ? JSON.parse(values.metadata) : {};
      } catch (e) {
        message.error("Invalid JSON in metadata field");
        return;
      }

      const updateData: any = {
        team_id: teamId,
        team_alias: values.team_alias,
        models: values.models,
        tpm_limit: values.tpm_limit,
        rpm_limit: values.rpm_limit,
        max_budget: values.max_budget,
        budget_duration: values.budget_duration,
        metadata: {
          ...parsedMetadata,
          guardrails: values.guardrails || []
        },
        organization_id: values.organization_id,
      };

      // Handle object_permission updates
      if (values.vector_stores !== undefined) {
        updateData.object_permission = {
          ...teamData?.team_info.object_permission,
          vector_stores: values.vector_stores || []
        };
      }
      
      const response = await teamUpdateCall(accessToken, updateData);
      
      message.success("Team settings updated successfully");
      setIsEditing(false);
      fetchTeamInfo();
    } catch (error) {
      console.error("Error updating team:", error);
    }
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

      <TabGroup defaultIndex={editTeam ? 3 : 0}>
        <TabList className="mb-4">
          {[
            <Tab key="overview">Overview</Tab>,
            ...(canEditTeam ? [
              <Tab key="members">Members</Tab>,
              <Tab key="member-permissions">Member Permissions</Tab>,
              <Tab key="settings">Settings</Tab>
            ] : [])
          ]}
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
                  {info.models.length === 0 ? (
                    <Badge color="red">All proxy models</Badge>
                  ) : (
                    info.models.map((model, index) => (
                      <Badge key={index} color="red">
                        {model}
                      </Badge>
                    ))
                  )}
                </div>
              </Card>

              <ObjectPermissionsView 
                objectPermission={info.object_permission} 
                variant="card"
                accessToken={accessToken}
              />
            </Grid>
          </TabPanel>

          {/* Members Panel */}
          <TabPanel>
            <TeamMembersComponent
              teamData={teamData}
              canEditTeam={canEditTeam}
              handleMemberDelete={handleMemberDelete}
              setSelectedEditMember={setSelectedEditMember}
              setIsEditMemberModalVisible={setIsEditMemberModalVisible}
              setIsAddMemberModalVisible={setIsAddMemberModalVisible}
            />
          </TabPanel>

          {/* Member Permissions Panel */}
          {canEditTeam && (
            <TabPanel>
              <MemberPermissions 
                teamId={teamId}
                accessToken={accessToken}
                canEditTeam={canEditTeam}
              />
            </TabPanel>
          )}

          {/* Settings Panel */}
          <TabPanel>
            <Card className="overflow-y-auto max-h-[65vh]">
              <div className="flex justify-between items-center mb-4">
                <Title>Team Settings</Title>
                {(canEditTeam && !isEditing) && (
                  <TremorButton 
                    onClick={() => setIsEditing(true)}
                  >
                    Edit Settings
                  </TremorButton>
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
                    guardrails: info.metadata?.guardrails || [],
                    metadata: info.metadata ? JSON.stringify(info.metadata, null, 2) : "",
                    organization_id: info.organization_id,
                    vector_stores: info.object_permission?.vector_stores || []
                  }}
                  layout="vertical"
                >
                  <Form.Item
                    label="Team Name"
                    name="team_alias"
                    rules={[{ required: true, message: "Please input a team name" }]}
                  >
                    <Input type=""/>
                  </Form.Item>
                  
                  <Form.Item label="Models" name="models">
                    <Select
                      mode="multiple"
                      placeholder="Select models"
                    >
                      <Select.Option key="all-proxy-models" value="all-proxy-models">
                        All Proxy Models
                      </Select.Option>
                      {userModels.map((model, idx) => (
                        <Select.Option key={idx} value={model}>
                          {getModelDisplayName(model)}
                        </Select.Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item label="Max Budget (USD)" name="max_budget">
                    <NumericalInput step={0.01} precision={2} style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item label="Reset Budget" name="budget_duration">
                    <Select placeholder="n/a">
                      <Select.Option value="24h">daily</Select.Option>
                      <Select.Option value="7d">weekly</Select.Option>
                      <Select.Option value="30d">monthly</Select.Option>
                    </Select>
                  </Form.Item>

                  <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
                    <NumericalInput step={1} style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
                    <NumericalInput step={1} style={{ width: "100%" }} />
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

                  <Form.Item label="Vector Stores" name="vector_stores">
                    <VectorStoreSelector
                      onChange={(values) => form.setFieldValue('vector_stores', values)}
                      value={form.getFieldValue('vector_stores')}
                      accessToken={accessToken || ""}
                      placeholder="Select vector stores"
                    />
                  </Form.Item>
                  
                  <Form.Item label="Organization ID" name="organization_id">
                    <Input type=""/>
                  </Form.Item>

                  <Form.Item label="Metadata" name="metadata">
                    <Input.TextArea rows={10} />
                  </Form.Item>
                  
                  <div className="sticky z-10 bg-white p-4 border-t border-gray-200 bottom-[-1.5rem] inset-x-[-1.5rem]">
                    <div className="flex justify-end items-center gap-2">
                      <Button htmlType="button" onClick={() => setIsEditing(false)}>
                        Cancel
                      </Button>
                      <TremorButton type="submit">
                        Save Changes
                      </TremorButton>
                    </div>
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
                      <div>Max: {info.max_budget !== null ? `$${info.max_budget}` : 'No Limit'}</div>
                    <div>Reset: {info.budget_duration || 'Never'}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Organization ID</Text>
                    <div>{info.organization_id}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Status</Text>
                    <Badge color={info.blocked ? 'red' : 'green'}>
                      {info.blocked ? 'Blocked' : 'Active'}
                    </Badge>
                  </div>

                  <ObjectPermissionsView 
                    objectPermission={info.object_permission} 
                    variant="inline"
                    className="pt-4 border-t border-gray-200"
                    accessToken={accessToken}
                  />
                </div>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <MemberModal
        visible={isEditMemberModalVisible}
        onCancel={() => setIsEditMemberModalVisible(false)}
        onSubmit={handleMemberUpdate}
        initialData={selectedEditMember}
        mode="edit"
        config={{
          title: "Edit Member",
          showEmail: true,
          showUserId: true,
          roleOptions: [
            { label: "Admin", value: "admin" },
            { label: "User", value: "user" }
          ]
        }}
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