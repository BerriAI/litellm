import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  TextInput,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Grid,
  Badge,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Button as TremorButton,
  Icon,
} from "@tremor/react";
import NumericalInput from "../shared/numerical_input";
import { Button, Form, Input, Select } from "antd";
import { ArrowLeftIcon, PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import {
  Member,
  Organization,
  organizationInfoCall,
  organizationMemberAddCall,
  organizationMemberUpdateCall,
  organizationMemberDeleteCall,
  organizationUpdateCall,
} from "../networking";
import UserSearchModal from "../common_components/user_search_modal";
import MemberModal from "../team/edit_membership";
import ObjectPermissionsView from "../object_permissions_view";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import { copyToClipboard as utilCopyToClipboard, formatNumberWithCommas } from "@/utils/dataUtils";
import { CheckIcon, CopyIcon } from "lucide-react";
import NotificationsManager from "../molecules/notifications_manager";

interface OrganizationInfoProps {
  organizationId: string;
  onClose: () => void;
  accessToken: string | null;
  is_org_admin: boolean;
  is_proxy_admin: boolean;
  userModels: string[];
  editOrg: boolean;
}

const OrganizationInfoView: React.FC<OrganizationInfoProps> = ({
  organizationId,
  onClose,
  accessToken,
  is_org_admin,
  is_proxy_admin,
  userModels,
  editOrg,
}) => {
  const [orgData, setOrgData] = useState<Organization | null>(null);
  const [loading, setLoading] = useState(true);
  const [form] = Form.useForm();
  const [isEditing, setIsEditing] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(null);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const canEditOrg = is_org_admin || is_proxy_admin;

  const fetchOrgInfo = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await organizationInfoCall(accessToken, organizationId);
      setOrgData(response);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to load organization information");
      console.error("Error fetching organization info:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOrgInfo();
  }, [organizationId, accessToken]);

  const handleMemberAdd = async (values: any) => {
    try {
      if (accessToken == null) {
        return;
      }

      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };
      const response = await organizationMemberAddCall(accessToken, organizationId, member);

      NotificationsManager.success("Organization member added successfully");
      setIsAddMemberModalVisible(false);
      form.resetFields();
      fetchOrgInfo();
    } catch (error) {
      NotificationsManager.fromBackend("Failed to add organization member");
      console.error("Error adding organization member:", error);
    }
  };

  const handleMemberUpdate = async (values: any) => {
    try {
      if (!accessToken) return;

      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };

      const response = await organizationMemberUpdateCall(accessToken, organizationId, member);
      NotificationsManager.success("Organization member updated successfully");
      setIsEditMemberModalVisible(false);
      form.resetFields();
      fetchOrgInfo();
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update organization member");
      console.error("Error updating organization member:", error);
    }
  };

  const handleMemberDelete = async (values: any) => {
    try {
      if (!accessToken) return;

      await organizationMemberDeleteCall(accessToken, organizationId, values.user_id);
      NotificationsManager.success("Organization member deleted successfully");
      setIsEditMemberModalVisible(false);
      form.resetFields();
      fetchOrgInfo();
    } catch (error) {
      NotificationsManager.fromBackend("Failed to delete organization member");
      console.error("Error deleting organization member:", error);
    }
  };

  const handleOrgUpdate = async (values: any) => {
    try {
      if (!accessToken) return;

      const updateData: any = {
        organization_id: organizationId,
        organization_alias: values.organization_alias,
        models: values.models,
        litellm_budget_table: {
          tpm_limit: values.tpm_limit,
          rpm_limit: values.rpm_limit,
          max_budget: values.max_budget,
          budget_duration: values.budget_duration,
        },
        metadata: values.metadata ? JSON.parse(values.metadata) : null,
      };

      // Handle object_permission updates
      if (values.vector_stores !== undefined || values.mcp_servers_and_groups !== undefined) {
        updateData.object_permission = {
          ...orgData?.object_permission,
          vector_stores: values.vector_stores || [],
        };

        if (values.mcp_servers_and_groups !== undefined) {
          const { servers, accessGroups } = values.mcp_servers_and_groups || {
            servers: [],
            accessGroups: [],
          };
          if (servers && servers.length > 0) {
            updateData.object_permission.mcp_servers = servers;
          }
          if (accessGroups && accessGroups.length > 0) {
            updateData.object_permission.mcp_access_groups = accessGroups;
          }
        }
      }

      const response = await organizationUpdateCall(accessToken, updateData);

      NotificationsManager.success("Organization settings updated successfully");
      setIsEditing(false);
      fetchOrgInfo();
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update organization settings");
      console.error("Error updating organization:", error);
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!orgData) {
    return <div className="p-4">Organization not found</div>;
  }

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  return (
    <div className="w-full h-screen p-4 bg-white">
      <div className="flex justify-between items-center mb-6">
        <div>
          <TremorButton icon={ArrowLeftIcon} onClick={onClose} variant="light" className="mb-4">
            Back to Organizations
          </TremorButton>
          <Title>{orgData.organization_alias}</Title>
          <div className="flex items-center cursor-pointer">
            <Text className="text-gray-500 font-mono">{orgData.organization_id}</Text>
            <Button
              type="text"
              size="small"
              icon={copiedStates["org-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(orgData.organization_id, "org-id")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["org-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
        </div>
      </div>

      <TabGroup defaultIndex={editOrg ? 2 : 0}>
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
                <Text>Organization Details</Text>
                <div className="mt-2">
                  <Text>Created: {new Date(orgData.created_at).toLocaleDateString()}</Text>
                  <Text>Updated: {new Date(orgData.updated_at).toLocaleDateString()}</Text>
                  <Text>Created By: {orgData.created_by}</Text>
                </div>
              </Card>

              <Card>
                <Text>Budget Status</Text>
                <div className="mt-2">
                  <Title>${formatNumberWithCommas(orgData.spend, 4)}</Title>
                  <Text>
                    of{" "}
                    {orgData.litellm_budget_table.max_budget === null
                      ? "Unlimited"
                      : `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`}
                  </Text>
                  {orgData.litellm_budget_table.budget_duration && (
                    <Text className="text-gray-500">Reset: {orgData.litellm_budget_table.budget_duration}</Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Rate Limits</Text>
                <div className="mt-2">
                  <Text>TPM: {orgData.litellm_budget_table.tpm_limit || "Unlimited"}</Text>
                  <Text>RPM: {orgData.litellm_budget_table.rpm_limit || "Unlimited"}</Text>
                  {orgData.litellm_budget_table.max_parallel_requests && (
                    <Text>Max Parallel Requests: {orgData.litellm_budget_table.max_parallel_requests}</Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Models</Text>
                <div className="mt-2 flex flex-wrap gap-2">
                  {orgData.models.length === 0 ? (
                    <Badge color="red">All proxy models</Badge>
                  ) : (
                    orgData.models.map((model, index) => (
                      <Badge key={index} color="red">
                        {model}
                      </Badge>
                    ))
                  )}
                </div>
              </Card>
              <Card>
                <Text>Teams</Text>
                <div className="mt-2 flex flex-wrap gap-2">
                  {orgData.teams?.map((team, index) => (
                    <Badge key={index} color="red">
                      {team.team_id}
                    </Badge>
                  ))}
                </div>
              </Card>

              <ObjectPermissionsView
                objectPermission={orgData.object_permission}
                variant="card"
                accessToken={accessToken}
              />
            </Grid>
          </TabPanel>

          {/* Budget Panel */}
          <TabPanel>
            <div className="space-y-4">
              <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[75vh]">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>User ID</TableHeaderCell>
                      <TableHeaderCell>Role</TableHeaderCell>
                      <TableHeaderCell>Spend</TableHeaderCell>
                      <TableHeaderCell>Created At</TableHeaderCell>
                      <TableHeaderCell></TableHeaderCell>
                    </TableRow>
                  </TableHead>

                  <TableBody>
                    {orgData.members?.map((member, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Text className="font-mono">{member.user_id}</Text>
                        </TableCell>
                        <TableCell>
                          <Text className="font-mono">{member.user_role}</Text>
                        </TableCell>
                        <TableCell>
                          <Text>${formatNumberWithCommas(member.spend, 4)}</Text>
                        </TableCell>
                        <TableCell>
                          <Text>{new Date(member.created_at).toLocaleString()}</Text>
                        </TableCell>
                        <TableCell>
                          {canEditOrg && (
                            <>
                              <Icon
                                icon={PencilAltIcon}
                                size="sm"
                                onClick={() => {
                                  setSelectedEditMember({
                                    role: member.user_role,
                                    user_email: member.user_email,
                                    user_id: member.user_id,
                                  });
                                  setIsEditMemberModalVisible(true);
                                }}
                              />
                              <Icon
                                icon={TrashIcon}
                                size="sm"
                                onClick={() => {
                                  handleMemberDelete(member);
                                }}
                              />
                            </>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Card>
              {canEditOrg && (
                <TremorButton
                  onClick={() => {
                    setIsAddMemberModalVisible(true);
                  }}
                >
                  Add Member
                </TremorButton>
              )}
            </div>
          </TabPanel>

          {/* Settings Panel */}
          <TabPanel>
            <Card className="overflow-y-auto max-h-[65vh]">
              <div className="flex justify-between items-center mb-4">
                <Title>Organization Settings</Title>
                {canEditOrg && !isEditing && (
                  <TremorButton onClick={() => setIsEditing(true)}>Edit Settings</TremorButton>
                )}
              </div>

              {isEditing ? (
                <Form
                  form={form}
                  onFinish={handleOrgUpdate}
                  initialValues={{
                    organization_alias: orgData.organization_alias,
                    models: orgData.models,
                    tpm_limit: orgData.litellm_budget_table.tpm_limit,
                    rpm_limit: orgData.litellm_budget_table.rpm_limit,
                    max_budget: orgData.litellm_budget_table.max_budget,
                    budget_duration: orgData.litellm_budget_table.budget_duration,
                    metadata: orgData.metadata ? JSON.stringify(orgData.metadata, null, 2) : "",
                    vector_stores: orgData.object_permission?.vector_stores || [],
                    mcp_servers_and_groups: {
                      servers: orgData.object_permission?.mcp_servers || [],
                      accessGroups: orgData.object_permission?.mcp_access_groups || [],
                    },
                  }}
                  layout="vertical"
                >
                  <Form.Item
                    label="Organization Name"
                    name="organization_alias"
                    rules={[
                      {
                        required: true,
                        message: "Please input an organization name",
                      },
                    ]}
                  >
                    <TextInput />
                  </Form.Item>

                  <Form.Item label="Models" name="models">
                    <Select mode="multiple" placeholder="Select models">
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

                  <Form.Item label="Vector Stores" name="vector_stores">
                    <VectorStoreSelector
                      onChange={(values) => form.setFieldValue("vector_stores", values)}
                      value={form.getFieldValue("vector_stores")}
                      accessToken={accessToken || ""}
                      placeholder="Select vector stores"
                    />
                  </Form.Item>

                  <Form.Item label="MCP Servers & Access Groups" name="mcp_servers_and_groups">
                    <MCPServerSelector
                      onChange={(values) => form.setFieldValue("mcp_servers_and_groups", values)}
                      value={form.getFieldValue("mcp_servers_and_groups")}
                      accessToken={accessToken || ""}
                      placeholder="Select MCP servers and access groups"
                    />
                  </Form.Item>

                  <Form.Item label="Metadata" name="metadata">
                    <Input.TextArea rows={4} />
                  </Form.Item>

                  <div className="sticky z-10 bg-white p-4 border-t border-gray-200 bottom-[-1.5rem] inset-x-[-1.5rem]">
                    <div className="flex justify-end items-center gap-2">
                      <Button onClick={() => setIsEditing(false)}>Cancel</Button>
                      <TremorButton type="submit">Save Changes</TremorButton>
                    </div>
                  </div>
                </Form>
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Organization Name</Text>
                    <div>{orgData.organization_alias}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Organization ID</Text>
                    <div className="font-mono">{orgData.organization_id}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Created At</Text>
                    <div>{new Date(orgData.created_at).toLocaleString()}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Models</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {orgData.models.map((model, index) => (
                        <Badge key={index} color="red">
                          {model}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Rate Limits</Text>
                    <div>TPM: {orgData.litellm_budget_table.tpm_limit || "Unlimited"}</div>
                    <div>RPM: {orgData.litellm_budget_table.rpm_limit || "Unlimited"}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Budget</Text>
                    <div>
                      Max:{" "}
                      {orgData.litellm_budget_table.max_budget !== null
                        ? `$${formatNumberWithCommas(orgData.litellm_budget_table.max_budget, 4)}`
                        : "No Limit"}
                    </div>
                    <div>Reset: {orgData.litellm_budget_table.budget_duration || "Never"}</div>
                  </div>

                  <ObjectPermissionsView
                    objectPermission={orgData.object_permission}
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
      <UserSearchModal
        isVisible={isAddMemberModalVisible}
        onCancel={() => setIsAddMemberModalVisible(false)}
        onSubmit={handleMemberAdd}
        accessToken={accessToken}
        title="Add Organization Member"
        roles={[
          {
            label: "org_admin",
            value: "org_admin",
            description: "Can add and remove members, and change their roles.",
          },
          {
            label: "internal_user",
            value: "internal_user",
            description: "Can view/create keys for themselves within organization.",
          },
          {
            label: "internal_user_viewer",
            value: "internal_user_viewer",
            description: "Can only view their keys within organization.",
          },
        ]}
        defaultRole="internal_user"
      />
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
            { label: "Org Admin", value: "org_admin" },
            { label: "Internal User", value: "internal_user" },
            { label: "Internal User Viewer", value: "internal_user_viewer" },
          ],
        }}
      />
    </div>
  );
};

export default OrganizationInfoView;
