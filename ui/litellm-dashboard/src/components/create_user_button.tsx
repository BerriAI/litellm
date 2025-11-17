import React, { useState, useEffect } from "react";
import { Button, Modal, Form, Input, Select, Select as Select2 } from "antd";
import {
  Button as Button2,
  Text,
  TextInput,
  SelectItem,
  Accordion,
  AccordionHeader,
  AccordionBody,
  Title,
} from "@tremor/react";
import OnboardingModal from "./onboarding_link";
import { InvitationLink } from "./onboarding_link";
import {
  userCreateCall,
  modelAvailableCall,
  invitationCreateCall,
  getProxyUISettings,
  getProxyBaseUrl,
} from "./networking";
import BulkCreateUsers from "./bulk_create_users_button";
const { Option } = Select;
import { Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { useQueryClient } from "@tanstack/react-query";
import NotificationsManager from "./molecules/notifications_manager";
import TeamDropdown from "./common_components/team_dropdown";

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

const Createuser: React.FC<CreateuserProps> = ({
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
  // get all models
  useEffect(() => {
    const fetchData = async () => {
      try {
        const userRole = "any"; // You may need to get the user role dynamically
        const modelDataResponse = await modelAvailableCall(accessToken, userID, userRole);
        // Assuming modelDataResponse.data contains an array of model objects with a 'model_name' property
        const availableModels = [];
        for (let i = 0; i < modelDataResponse.data.length; i++) {
          const model = modelDataResponse.data[i];
          availableModels.push(model.id);
        }
        console.log("Model data response:", modelDataResponse.data);
        console.log("Available models:", availableModels);

        // Assuming modelDataResponse.data contains an array of model names
        setUserModels(availableModels);

        // get ui settings
        const uiSettingsResponse = await getProxyUISettings(accessToken);
        console.log("uiSettingsResponse:", uiSettingsResponse);

        setUISettings(uiSettingsResponse);
      } catch (error) {
        console.error("Error fetching model data:", error);
      }
    };

    setBaseUrl(getProxyBaseUrl());

    fetchData(); // Call the function to fetch model data when the component mounts
  }, []); // Empty dependency array to run only once

  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiuser(false);
    form.resetFields();
  };

  const handleCreate = async (formValues: { user_id: string; models?: string[]; user_role: string }) => {
    try {
      NotificationsManager.info("Making API Call");
      if (!isEmbedded) {
        setIsModalVisible(true);
      }
      if ((!formValues.models || formValues.models.length === 0) && formValues.user_role !== "proxy_admin") {
        console.log("formValues.user_role", formValues.user_role);
        // If models is empty or undefined, set it to "no-default-models"
        formValues.models = ["no-default-models"];
      }
      console.log("formValues in create user:", formValues);
      const response = await userCreateCall(accessToken, null, formValues);
      await queryClient.invalidateQueries({ queryKey: ["userList"] });
      console.log("user create Response:", response);
      setApiuser(true);
      const user_id = response.data?.user_id || response.user_id;

      // Call the callback if provided (for embedded mode)
      if (onUserCreated && isEmbedded) {
        onUserCreated(user_id);
        form.resetFields();
        return; // Skip the invitation flow when embedded
      }

      // only do invite link flow if sso is not enabled
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
      localStorage.removeItem("userData" + userID);
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error?.message || "Error creating the user";
      NotificationsManager.fromBackend(errorMessage);
      console.error("Error creating the user:", error);
    }
  };

  // Modify the return statement to handle embedded mode
  if (isEmbedded) {
    return (
      <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
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
                    <p className="ml-2" style={{ color: "gray", fontSize: "12px" }}>
                      {description}
                    </p>
                  </div>
                </SelectItem>
              ))}
          </Select2>
        </Form.Item>
        <Form.Item label="Team" name="team_id">
          <Select placeholder="Select Team" style={{ width: "100%" }}>
            <TeamDropdown teams={teams} />
          </Select>
        </Form.Item>

        <Form.Item label="Metadata" name="metadata">
          <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
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
      <Button2 className="mb-0" onClick={() => setIsModalVisible(true)}>
        + Invite User
      </Button2>
      <BulkCreateUsers accessToken={accessToken} teams={teams} possibleUIRoles={possibleUIRoles} />
      <Modal
        title="Invite User"
        visible={isModalVisible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Text className="mb-1">Create a User who can own keys</Text>
        <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
          <Form.Item label="User Email" name="user_email">
            <TextInput placeholder="" />
          </Form.Item>
          <Form.Item
            label={
              <span>
                Global Proxy Role{" "}
                <Tooltip title="This is the role that the user will globally on the proxy. This role is independent of any team/org specific roles.">
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
                    <div className="flex">
                      {ui_label}{" "}
                      <p className="ml-2" style={{ color: "gray", fontSize: "12px" }}>
                        {description}
                      </p>
                    </div>
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
            <TeamDropdown teams={teams} />
          </Form.Item>

          <Form.Item label="Metadata" name="metadata">
            <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
          </Form.Item>
          <Accordion>
            <AccordionHeader>
              <Title>Personal Key Creation</Title>
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
            <Button htmlType="submit">Create User</Button>
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

export default Createuser;
