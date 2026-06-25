import { InfoCircleOutlined, UserAddOutlined } from "@ant-design/icons";
import { useQueryClient } from "@tanstack/react-query";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { Accordion, AccordionBody, AccordionHeader, SelectItem, TextInput } from "@tremor/react";
import {
  Alert,
  Button,
  Checkbox,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Select as Select2,
  Space,
  Switch,
  Tooltip,
  Typography,
} from "antd";
import debounce from "lodash/debounce";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import BulkCreateUsers from "./bulk_create_users_button";
import TeamDropdown from "./common_components/team_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import NotificationsManager from "./molecules/notifications_manager";
import {
  getProxyBaseUrl,
  getProxyUISettings,
  invitationCreateCall,
  modelAvailableCall,
  userCreateCall,
  userFilterUICall,
} from "./networking";
import OnboardingModal, { InvitationLink } from "./onboarding_link";
const { Option } = Select;
const { Text, Link, Title } = Typography;
// Helper function to generate UUID compatible across all environments
const generateUUID = (): string => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback UUID generation for environments without crypto.randomUUID
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c == "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

interface CreateuserProps {
  userID: string;
  accessToken: string;
  teams: any[] | null;
  possibleUIRoles: null | Record<string, Record<string, string>>;
  onUserCreated?: (userId: string) => void;
  isEmbedded?: boolean;
}

// Define an interface for the UI settings
interface UISettings {
  PROXY_BASE_URL: string | null;
  PROXY_LOGOUT_URL: string | null;
  DEFAULT_TEAM_DISABLED: boolean;
  SSO_ENABLED: boolean;
}

export const CreateUserButton: React.FC<CreateuserProps> = ({
  userID,
  accessToken,
  teams,
  possibleUIRoles,
  onUserCreated,
  isEmbedded = false,
}) => {
  const queryClient = useQueryClient();
  const [uiSettings, setUISettings] = useState<UISettings | null>(null);
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiuser, setApiuser] = useState<boolean>(false);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] = useState(false);
  const [invitationLinkData, setInvitationLinkData] = useState<InvitationLink | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const { data: organizations = [] } = useOrganizations();
  const [ownerOptions, setOwnerOptions] = useState<{ label: string; value: string }[]>([]);
  const [ownerSearchLoading, setOwnerSearchLoading] = useState(false);
  const [isServiceAccount, setIsServiceAccount] = useState(false);

  const fetchOwners = async (searchText: string) => {
    if (!searchText) {
      setOwnerOptions([]);
      return;
    }
    setOwnerSearchLoading(true);
    try {
      const params = new URLSearchParams();
      params.append("user_email", searchText);
      const results = await userFilterUICall(accessToken, params);
      setOwnerOptions(
        results.map((u: { user_email: string; user_id: string }) => ({
          label: `${u.user_email} (${u.user_id})`,
          value: u.user_id,
        }))
      );
    } catch {
      // silently ignore search errors
    } finally {
      setOwnerSearchLoading(false);
    }
  };

  const debouncedOwnerSearch = useCallback(debounce(fetchOwners, 300), [accessToken]);

  // Derive teams from the user's organizations, falling back to the teams prop
  const availableTeams = useMemo(() => {
    const orgTeams = organizations.flatMap((org) => org.teams || []);
    if (orgTeams.length > 0) return orgTeams;
    return teams || [];
  }, [organizations, teams]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const userRole = "any";
        const modelDataResponse = await modelAvailableCall(accessToken, userID, userRole);
        const availableModels = [];
        for (let i = 0; i < modelDataResponse.data.length; i++) {
          const model = modelDataResponse.data[i];
          availableModels.push(model.id);
        }
        setUserModels(availableModels);
        const uiSettingsResponse = await getProxyUISettings(accessToken);
        setUISettings(uiSettingsResponse);
      } catch (error) {
        console.error("Error fetching model data:", error);
      }
    };

    setBaseUrl(getProxyBaseUrl());
    fetchData();
  }, []);

  const handleOk = () => {
    setIsModalVisible(false);
    setIsServiceAccount(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiuser(false);
    setIsServiceAccount(false);
    form.resetFields();
  };

  const handleCreate = async (formValues: {
    user_id: string;
    models?: string[];
    user_role: string;
    organization_ids?: string[];
    organizations?: string[];
    send_invite_email?: boolean;
    owner_ids?: string[];
    is_service_account?: boolean;
    name?: string;
    requested_models?: string[];
    use_case?: string;
    requested_rpm_limit?: number;
    requested_parallel_requests_limit?: number;
  }) => {
    try {
      NotificationsManager.info("Making API Call");
      if (!isEmbedded) {
        setIsModalVisible(true);
      }
      if ((!formValues.models || formValues.models.length === 0) && formValues.user_role !== "proxy_admin") {
        formValues.models = ["no-default-models"];
      }
      if (formValues.organization_ids) {
        formValues.organizations = formValues.organization_ids;
        delete formValues.organization_ids;
      }
      const response = await userCreateCall(accessToken, null, formValues);
      await queryClient.invalidateQueries({ queryKey: ["userList"] });
      setApiuser(true);
      const user_id = response.data?.user_id || response.user_id;

      if (onUserCreated && isEmbedded) {
        onUserCreated(user_id);
        form.resetFields();
        setIsServiceAccount(false);
        return;
      }

      if (!uiSettings?.SSO_ENABLED) {
        invitationCreateCall(accessToken, user_id).then((data) => {
          data.has_user_setup_sso = false;
          setInvitationLinkData(data);
          setIsInvitationLinkModalVisible(true);
        });
      } else {
        // create an InvitationLink Object for this user for the SSO flow
        // for SSO the invite link is the proxy base url since the User just needs to login
        const invitationLink: InvitationLink = {
          id: generateUUID(), // Generate a unique ID
          user_id: user_id,
          is_accepted: false,
          accepted_at: null,
          expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // Set expiry to 7 days from now
          created_at: new Date(),
          created_by: userID, // Assuming userID is the current user creating the invitation
          updated_at: new Date(),
          updated_by: userID,
          has_user_setup_sso: true,
        };
        setInvitationLinkData(invitationLink);
        setIsInvitationLinkModalVisible(true);
      }

      NotificationsManager.success("API user Created");
      form.resetFields();
      setIsServiceAccount(false);
      localStorage.removeItem("userData" + userID);
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error?.message || "Error creating the user";
      NotificationsManager.fromBackend(errorMessage);
      console.error("Error creating the user:", error);
    }
  };

  // Service account toggle. When on, the owners field (required, min 2) and
  // the extra service-account fields (name, requested_models, use_case, limits)
  // are revealed. When off, none of these render. These Form.Items are shared
  // by both the embedded and standalone (modal) forms to avoid duplication.
  const serviceAccountFields = (
    <>
      <Form.Item
        label={
          <span>
            Service Account{" "}
            <Tooltip title="Create a service account entry owned by the selected owners.">
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="is_service_account"
        valuePropName="checked"
        help="When on, this user is registered as a service account."
      >
        <Switch onChange={(v) => setIsServiceAccount(v)} />
      </Form.Item>

      {isServiceAccount && (
        <>
          <Form.Item
            label={
              <span>
                Owners{" "}
                <Tooltip title="At least 2 users who own this service account">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="owner_ids"
            required
            rules={[
              {
                validator: async (_, value) => {
                  if (!value || value.length < 2) {
                    throw new Error("A service account requires at least 2 owners");
                  }
                },
              },
            ]}
            help="required — at least 2 owners for this service account"
          >
            <Select2
              mode="multiple"
              showSearch
              placeholder="Search by email to add owners"
              filterOption={false}
              onSearch={(v) => debouncedOwnerSearch(v)}
              loading={ownerSearchLoading}
              options={ownerOptions}
              notFoundContent={ownerSearchLoading ? "Searching..." : "No users found"}
              style={{ width: "100%" }}
            />
          </Form.Item>

          <Form.Item label="Name" name="name">
            <Input placeholder="Service account name" />
          </Form.Item>
          <Form.Item
            label={
              <span>
                Requested Models{" "}
                <Tooltip title="Model public names this service account should be granted. Suggestions come from the proxy's available models.">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="requested_models"
            help="Start typing to see available model public names."
          >
            <Select2
              mode="tags"
              showSearch
              placeholder="Search for model public names"
              style={{ width: "100%" }}
            >
              {userModels.map((model) => (
                <Select2.Option key={model} value={model}>
                  {getModelDisplayName(model)}
                </Select2.Option>
              ))}
            </Select2>
          </Form.Item>
          <Form.Item label="Use Case" name="use_case">
            <Input.TextArea rows={2} placeholder="Describe the intended use case for this service account" />
          </Form.Item>
          <Form.Item label="Requested RPM Limit" name="requested_rpm_limit">
            <InputNumber min={0} placeholder="Requests per minute" style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item
            label="Requested Parallel Requests Limit"
            name="requested_parallel_requests_limit"
          >
            <InputNumber min={0} placeholder="Max parallel requests" style={{ width: "100%" }} />
          </Form.Item>
        </>
      )}
    </>
  );

  // Modify the return statement to handle embedded mode
  if (isEmbedded) {
    return (
      <Form
        form={form}
        onFinish={handleCreate}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        initialValues={{ user_role: "internal_user_viewer", send_invite_email: true }}
      >
        <Alert
          message="Email invitations"
          description={
            <>
              New users receive an email invite only when an email integration (SMTP, Resend, or SendGrid) is
              configured.{" "}
              <Link href="https://docs.litellm.ai/docs/proxy/email" target="_blank">
                Learn how to set up email notifications
              </Link>
            </>
          }
          type="info"
          showIcon
          className="mb-4"
        />
        <Form.Item label="User Email" name="user_email">
          <TextInput placeholder="" />
        </Form.Item>
        <Form.Item label="User Role" name="user_role">
          <Select2>
            {possibleUIRoles &&
              Object.entries(possibleUIRoles).map(([role, { ui_label, description }]) => (
                <SelectItem key={role} value={role} title={ui_label}>
                  <div className="flex">
                    {ui_label}{" "}
                    <Text className="ml-2" style={{ color: "gray", fontSize: "12px" }}>
                      {description}
                    </Text>
                  </div>
                </SelectItem>
              ))}
          </Select2>
        </Form.Item>
        <Form.Item label="Team" name="team_id">
          <TeamDropdown />
        </Form.Item>

        {serviceAccountFields}

        <Form.Item label="Metadata" name="metadata">
          <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
        </Form.Item>

        <Form.Item label="Send invitation email" name="send_invite_email" valuePropName="checked">
          <Checkbox />
        </Form.Item>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button htmlType="submit">Create User</Button>
        </div>
      </Form>
    );
  }

  // Original return for standalone mode
  return (
    <div className="flex gap-2">
      <Button type="primary" className="mb-0" onClick={() => setIsModalVisible(true)}>
        + Invite User
      </Button>
      <BulkCreateUsers accessToken={accessToken} teams={teams} possibleUIRoles={possibleUIRoles} />
      <Modal
        title="Invite User"
        open={isModalVisible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Space direction="vertical" size="middle">
          <Text className="mb-1">Create a User who can own keys</Text>
          <Alert
            message="Email invitations"
            description={
              <>
                New users receive an email invite only when an email integration (SMTP, Resend, or SendGrid) is
                configured.{" "}
                <Link href="https://docs.litellm.ai/docs/proxy/email" target="_blank">
                  Learn how to set up email notifications
                </Link>
              </>
            }
            type="info"
            showIcon
            className="mb-4"
          />
        </Space>
        <Form
          form={form}
          onFinish={handleCreate}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
          initialValues={{ user_role: "internal_user_viewer", send_invite_email: true }}
        >
          <Form.Item label="User Email" name="user_email">
            <Input />
          </Form.Item>
          <Form.Item
            label={
              <span>
                Global Proxy Role{" "}
                <Tooltip title="This role is independent of any team/org specific roles. Configure Team / Organization Admins in the Settings">
                  <InfoCircleOutlined />
                </Tooltip>
              </span>
            }
            name="user_role"
          >
            <Select2>
              {possibleUIRoles &&
                Object.entries(possibleUIRoles).map(([role, { ui_label, description }]) => (
                  <SelectItem key={role} value={role} title={ui_label}>
                    <Text>{ui_label}</Text>
                    <Text type="secondary">
                      {" - "}
                      {description}
                    </Text>
                  </SelectItem>
                ))}
            </Select2>
          </Form.Item>

          <Form.Item
            label="Team"
            className="gap-2"
            name="team_id"
            help="If selected, user will be added as a 'user' role to the team."
          >
            <TeamDropdown />
          </Form.Item>

          <Form.Item
            label="Organization"
            name="organization_ids"
            help="The user will be added to the selected organization(s)."
          >
            <Select mode="multiple" placeholder="Select Organization" style={{ width: "100%" }}>
              {organizations.map((org) => (
                <Option key={org.organization_id} value={org.organization_id}>
                  {org.organization_alias} ({org.organization_id})
                </Option>
              ))}
            </Select>
          </Form.Item>

          {serviceAccountFields}

          <Form.Item label="Metadata" name="metadata">
            <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
          </Form.Item>
          <Form.Item label="Send invitation email" name="send_invite_email" valuePropName="checked">
            <Checkbox />
          </Form.Item>
          <Accordion>
            <AccordionHeader>
              <Text strong>Personal Key Creation</Text>
            </AccordionHeader>
            <AccordionBody>
              <Form.Item
                className="gap-2"
                label={
                  <span>
                    Models{" "}
                    <Tooltip title="Models user has access to, outside of team scope.">
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="models"
                help="Models user has access to, outside of team scope."
              >
                <Select2 mode="multiple" placeholder="Select models" style={{ width: "100%" }}>
                  <Select2.Option key="all-proxy-models" value="all-proxy-models">
                    All Proxy Models
                  </Select2.Option>
                  <Select2.Option key="no-default-models" value="no-default-models">
                    No Default Models
                  </Select2.Option>
                  {userModels.map((model) => (
                    <Select2.Option key={model} value={model}>
                      {getModelDisplayName(model)}
                    </Select2.Option>
                  ))}
                </Select2>
              </Form.Item>
            </AccordionBody>
          </Accordion>

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button type="primary" icon={<UserAddOutlined />} htmlType="submit">
              Invite User
            </Button>
          </div>
        </Form>
      </Modal>
      {apiuser && (
        <OnboardingModal
          isInvitationLinkModalVisible={isInvitationLinkModalVisible}
          setIsInvitationLinkModalVisible={setIsInvitationLinkModalVisible}
          baseUrl={baseUrl || ""}
          invitationLinkData={invitationLinkData}
        />
      )}
    </div>
  );
};
