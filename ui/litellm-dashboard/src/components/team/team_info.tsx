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
  Icon,
  TextInput,
} from "@tremor/react";
import TeamMembersComponent from "./team_member_view";
import MemberPermissions from "./member_permissions";
import {
  teamInfoCall,
  teamMemberDeleteCall,
  teamMemberAddCall,
  teamMemberUpdateCall,
  Member,
  teamUpdateCall,
} from "@/components/networking";
import { Button, Form, Input, Select, message, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Select as Select2 } from "antd";
import {
  ArrowLeftIcon,
  PencilAltIcon,
  PlusIcon,
  TrashIcon,
} from "@heroicons/react/outline";
import MemberModal from "./edit_membership";
import UserSearchModal from "@/components/common_components/user_search_modal";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { isAdminRole } from "@/utils/roles";
import ObjectPermissionsView from "../object_permissions_view";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import PremiumVectorStoreSelector from "../common_components/PremiumVectorStoreSelector";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import EditLoggingSettings from "./EditLoggingSettings";
import LoggingSettingsView from "../logging_settings_view";
import { fetchMCPAccessGroups } from "../networking";
import { CheckIcon, CopyIcon } from "lucide-react";
import { copyToClipboard as utilCopyToClipboard } from "../../utils/dataUtils"

export interface TeamMembership {
  user_id: string;
  team_id: string;
  budget_id: string;
  spend: number;
  litellm_budget_table: {
    budget_id: string;
    soft_budget: number | null;
    max_budget: number | null;
    max_parallel_requests: number | null;
    tpm_limit: number | null;
    rpm_limit: number | null;
    model_max_budget: Record<string, number> | null;
    budget_duration: string | null;
  };
}

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
      mcp_access_groups?: string[];
      vector_stores: string[];
    };
    team_member_budget_table: {
      max_budget: number;
      budget_duration: string;
    } | null;
  };
  keys: any[];
  team_memberships: TeamMembership[];
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
  premiumUser = false,
  onUpdate,
}) => {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] =
    useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(
    null
  );
  const [isEditing, setIsEditing] = useState(false);
  const [mcpAccessGroups, setMcpAccessGroups] = useState<string[]>([]);
  const [mcpAccessGroupsLoaded, setMcpAccessGroupsLoaded] = useState(false);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({})

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

  const fetchMcpAccessGroups = async () => {
    if (!accessToken) return;
    if (mcpAccessGroupsLoaded) return;
    try {
      const groups = await fetchMCPAccessGroups(accessToken);
      setMcpAccessGroups(groups);
      setMcpAccessGroupsLoaded(true);
    } catch (error) {
      console.error("Failed to fetch MCP access groups:", error);
    }
  };

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

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
    } catch (error: any) {
      let errMsg = "Failed to add team member";

      if (
        error?.raw?.detail?.error?.includes(
          "Assigning team admins is a premium feature"
        )
      ) {
        errMsg =
          "Assigning admins is an enterprise-only feature. Please upgrade your LiteLLM plan to enable this.";
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
      };
      message.destroy(); // Remove all existing toasts

      await teamMemberUpdateCall(accessToken, teamId, member);

      message.success("Team member updated successfully");
      setIsEditMemberModalVisible(false);

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
    } catch (error: any) {
      let errMsg = "Failed to update team member";
      if (
        error?.raw?.detail?.includes(
          "Assigning team admins is a premium feature"
        )
      ) {
        errMsg =
          "Assigning admins is an enterprise-only feature. Please upgrade your LiteLLM plan to enable this.";
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

      await teamMemberDeleteCall(accessToken, teamId, member);

      message.success("Team member removed successfully");

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
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
          guardrails: values.guardrails || [],
          logging: values.logging_settings || [],
        },
        organization_id: values.organization_id,
      };

      if (values.team_member_budget !== undefined) {
        updateData.team_member_budget = Number(values.team_member_budget);
      }

      if (values.team_member_key_duration !== undefined) {
        updateData.team_member_key_duration = values.team_member_key_duration;
      }

      // Handle object_permission updates
      const { servers, accessGroups } = values.mcp_servers_and_groups || {
        servers: [],
        accessGroups: [],
      };
      if (
        (servers && servers.length > 0) ||
        (accessGroups && accessGroups.length > 0)
      ) {
        updateData.object_permission = {};
        if (servers && servers.length > 0) {
          updateData.object_permission.mcp_servers = servers;
        }
        if (accessGroups && accessGroups.length > 0) {
          updateData.object_permission.mcp_access_groups = accessGroups;
        }
      }
      delete values.mcp_servers_and_groups;

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

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text)
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }))
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }))
      }, 2000)
    }
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <TremorButton
            icon={ArrowLeftIcon}
            variant="light"
            onClick={onClose}
            className="mb-4"
          >
            Back to Teams
          </TremorButton>
          <Title>{info.team_alias}</Title>
          <div className="flex items-center cursor-pointer">
            <Text className="text-gray-500 font-mono">{info.team_id}</Text> 
            <Button
              type="text"
              size="small"
              icon={copiedStates["team-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(info.team_id, "team-id")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["team-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
        </div>
      </div>

      <TabGroup defaultIndex={editTeam ? 3 : 0}>
        <TabList className="mb-4">
          {[
            <Tab key="overview">Overview</Tab>,
            ...(canEditTeam
              ? [
                  <Tab key="members">Members</Tab>,
                  <Tab key="member-permissions">Member Permissions</Tab>,
                  <Tab key="settings">Settings</Tab>,
                ]
              : []),
          ]}
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Budget Status</Text>
                <div className="mt-2">
                  <Title>${formatNumberWithCommas(info.spend, 4)}</Title>
                  <Text>
                    of{" "}
                    {info.max_budget === null
                      ? "Unlimited"
                      : `$${formatNumberWithCommas(info.max_budget, 4)}`}
                  </Text>
                  {info.budget_duration && (
                    <Text className="text-gray-500">
                      Reset: {info.budget_duration}
                    </Text>
                  )}
                  <br />
                  {info.team_member_budget_table && (
                    <Text className="text-gray-500">
                      Team Member Budget: $
                      {formatNumberWithCommas(
                        info.team_member_budget_table.max_budget,
                        4
                      )}
                    </Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Rate Limits</Text>
                <div className="mt-2">
                  <Text>TPM: {info.tpm_limit || "Unlimited"}</Text>
                  <Text>RPM: {info.rpm_limit || "Unlimited"}</Text>
                  {info.max_parallel_requests && (
                    <Text>
                      Max Parallel Requests: {info.max_parallel_requests}
                    </Text>
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

              <Card>
                <Text className="font-semibold text-gray-900">Virtual Keys</Text>
                <div className="mt-2">
                    <Text>
                      User Keys: {teamData.keys.filter(key => key.user_id).length}
                    </Text>
                    <Text>
                      Service Account Keys: {teamData.keys.filter(key => !key.user_id).length}
                    </Text>
                    <Text className="text-gray-500">
                      Total: {teamData.keys.length}
                    </Text>
                  </div>
              </Card>

              <ObjectPermissionsView
                objectPermission={info.object_permission}
                variant="card"
                accessToken={accessToken}
              />

              <LoggingSettingsView
                loggingConfigs={info.metadata?.logging || []}
                disabledCallbacks={[]}
                variant="card"
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
                {canEditTeam && !isEditing && (
                  <TremorButton onClick={() => setIsEditing(true)}>
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
                    metadata: info.metadata
                      ? JSON.stringify(
                          (({ logging, ...rest }) => rest)(info.metadata),
                          null,
                          2
                        )
                      : "",
                    logging_settings: info.metadata?.logging || [],
                    organization_id: info.organization_id,
                    vector_stores: info.object_permission?.vector_stores || [],
                    mcp_servers: info.object_permission?.mcp_servers || [],
                    mcp_access_groups: info.object_permission?.mcp_access_groups || [],
                    mcp_servers_and_groups: {
                      servers: info.object_permission?.mcp_servers || [],
                      accessGroups: info.object_permission?.mcp_access_groups || [],
                    },
                  }}
                  layout="vertical"
                >
                  <Form.Item
                    label="Team Name"
                    name="team_alias"
                    rules={[
                      { required: true, message: "Please input a team name" },
                    ]}
                  >
                    <Input type="" />
                  </Form.Item>

                  <Form.Item label="Models" name="models">
                    <Select mode="multiple" placeholder="Select models">
                      <Select.Option
                        key="all-proxy-models"
                        value="all-proxy-models"
                      >
                        All Proxy Models
                      </Select.Option>
                      {Array.from(new Set(userModels)).map((model, idx) => (
                        <Select.Option key={idx} value={model}>
                          {getModelDisplayName(model)}
                        </Select.Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item label="Max Budget (USD)" name="max_budget">
                    <NumericalInput
                      step={0.01}
                      precision={2}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>

                  <Form.Item
                    label="Team Member Budget (USD)"
                    name="team_member_budget"
                    tooltip="This is the individual budget for a user in the team."
                  >
                    <NumericalInput
                      step={0.01}
                      precision={2}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>

                  <Form.Item
                    label="Team Member Key Duration (eg: 1d, 1mo)"
                    name="team_member_key_duration"
                    tooltip="Set a limit to the duration of a team member's key. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days), 1mo (month)"
                  >
                    <TextInput placeholder="e.g., 30d" />
                  </Form.Item>

                  <Form.Item label="Reset Budget" name="budget_duration">
                    <Select placeholder="n/a">
                      <Select.Option value="24h">daily</Select.Option>
                      <Select.Option value="7d">weekly</Select.Option>
                      <Select.Option value="30d">monthly</Select.Option>
                    </Select>
                  </Form.Item>

                  <Form.Item
                    label="Tokens per minute Limit (TPM)"
                    name="tpm_limit"
                  >
                    <NumericalInput step={1} style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item
                    label="Requests per minute Limit (RPM)"
                    name="rpm_limit"
                  >
                    <NumericalInput step={1} style={{ width: "100%" }} />
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
                    help="Select existing guardrails or enter new ones"
                  >
                    <Select
                      mode="tags"
                      placeholder="Select or enter guardrails"
                    />
                  </Form.Item>

                  <Form.Item label="Vector Stores" name="vector_stores">
                    <VectorStoreSelector
                      onChange={(values) =>
                        form.setFieldValue("vector_stores", values)
                      }
                      value={form.getFieldValue("vector_stores")}
                      accessToken={accessToken || ""}
                      placeholder="Select vector stores"
                    />
                  </Form.Item>

                  <Form.Item
                    label="MCP Servers / Access Groups"
                    name="mcp_servers_and_groups"
                  >
                    <MCPServerSelector
                      onChange={(val) =>
                        form.setFieldValue("mcp_servers_and_groups", val)
                      }
                      value={form.getFieldValue("mcp_servers_and_groups")}
                      accessToken={accessToken || ""}
                      placeholder="Select MCP servers or access groups (optional)"
                    />
                  </Form.Item>
                  <Form.Item label="Organization ID" name="organization_id">
                    <Input type="" />
                  </Form.Item>

                  <Form.Item label="Logging Settings" name="logging_settings">
                    <EditLoggingSettings
                      value={form.getFieldValue("logging_settings")}
                      onChange={(values) =>
                        form.setFieldValue("logging_settings", values)
                      }
                    />
                  </Form.Item>

                  <Form.Item label="Metadata" name="metadata">
                    <Input.TextArea rows={10} />
                  </Form.Item>

                  <div className="sticky z-10 bg-white p-4 border-t border-gray-200 bottom-[-1.5rem] inset-x-[-1.5rem]">
                    <div className="flex justify-end items-center gap-2">
                      <Button
                        htmlType="button"
                        onClick={() => setIsEditing(false)}
                      >
                        Cancel
                      </Button>
                      <TremorButton type="submit">Save Changes</TremorButton>
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
                    <div>TPM: {info.tpm_limit || "Unlimited"}</div>
                    <div>RPM: {info.rpm_limit || "Unlimited"}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Team Budget</Text>
                    <div>
                      Max Budget:{" "}
                      {info.max_budget !== null
                        ? `$${formatNumberWithCommas(info.max_budget, 4)}`
                        : "No Limit"}
                    </div>
                    <div>Budget Reset: {info.budget_duration || "Never"}</div>
                  </div>
                  <div>
                    <Text className="font-medium">
                      Team Member Settings{" "}
                      <Tooltip title="These are limits on individual team members">
                        <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                      </Tooltip>
                    </Text>
                    <div>
                      Max Budget:{" "}
                      {info.team_member_budget_table?.max_budget || "No Limit"}
                    </div>
                    <div>
                      Key Duration:{" "}
                      {info.metadata?.team_member_key_duration || "No Limit"}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Organization ID</Text>
                    <div>{info.organization_id}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Status</Text>
                    <Badge color={info.blocked ? "red" : "green"}>
                      {info.blocked ? "Blocked" : "Active"}
                    </Badge>
                  </div>

                  <ObjectPermissionsView
                    objectPermission={info.object_permission}
                    variant="inline"
                    className="pt-4 border-t border-gray-200"
                    accessToken={accessToken}
                  />

                  <LoggingSettingsView
                    loggingConfigs={info.metadata?.logging || []}
                    disabledCallbacks={[]}
                    variant="inline"
                    className="pt-4 border-t border-gray-200"
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
            { label: "User", value: "user" },
          ],
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
