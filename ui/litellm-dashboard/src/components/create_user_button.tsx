import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Button,
  Modal,
  Form,
  Input,
  message,
  Select,
  InputNumber,
  Select as Select2,
} from "antd";
import { Button as Button2, Text, TextInput, SelectItem } from "@tremor/react";
import OnboardingModal from "./onboarding_link";
import { InvitationLink } from "./onboarding_link";
import {
  userCreateCall,
  modelAvailableCall,
  invitationCreateCall,
} from "./networking";
const { Option } = Select;

interface CreateuserProps {
  userID: string;
  accessToken: string;
  teams: any[] | null;
  possibleUIRoles: null | Record<string, Record<string, string>>;
}

const Createuser: React.FC<CreateuserProps> = ({
  userID,
  accessToken,
  teams,
  possibleUIRoles,
}) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiuser, setApiuser] = useState<string | null>(null);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] =
    useState(false);
  const [invitationLinkData, setInvitationLinkData] =
    useState<InvitationLink | null>(null);
  const router = useRouter();
  const isLocal = process.env.NODE_ENV === "development";
  const [baseUrl, setBaseUrl] = useState(
    isLocal ? "http://localhost:4000" : ""
  );
  // get all models
  useEffect(() => {
    const fetchData = async () => {
      try {
        const userRole = "any"; // You may need to get the user role dynamically
        const modelDataResponse = await modelAvailableCall(
          accessToken,
          userID,
          userRole
        );
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
      } catch (error) {
        console.error("Error fetching model data:", error);
      }
    };

    fetchData(); // Call the function to fetch model data when the component mounts
  }, []); // Empty dependency array to run only once

  useEffect(() => {
    if (router) {
      const { protocol, host } = window.location;
      const baseUrl = `${protocol}/${host}`;
      setBaseUrl(baseUrl);
    }
  }, [router]);
  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiuser(null);
    form.resetFields();
  };

  const handleCreate = async (formValues: { user_id: string }) => {
    try {
      message.info("Making API Call");
      setIsModalVisible(true);
      console.log("formValues in create user:", formValues);
      const response = await userCreateCall(accessToken, null, formValues);
      console.log("user create Response:", response);
      setApiuser(response["key"]);
      const user_id = response.data?.user_id || response.user_id;
      invitationCreateCall(accessToken, user_id).then((data) => {
        setInvitationLinkData(data);
        setIsInvitationLinkModalVisible(true);
      });
      message.success("API user Created");
      form.resetFields();
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      console.error("Error creating the user:", error);
    }
  };

  return (
    <div>
      <Button2 className="mx-auto mb-0" onClick={() => setIsModalVisible(true)}>
        + Invite User
      </Button2>
      <Modal
        title="Invite User"
        visible={isModalVisible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Text className="mb-1">Create a User who can own keys</Text>
        <Form
          form={form}
          onFinish={handleCreate}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <Form.Item label="User Email" name="user_email">
            <TextInput placeholder="" />
          </Form.Item>
          <Form.Item label="User Role" name="user_role">
            <Select2>
              {possibleUIRoles &&
                Object.entries(possibleUIRoles).map(
                  ([role, { ui_label, description }]) => (
                    <SelectItem key={role} value={role} title={ui_label}>
                      <div className="flex">
                        {ui_label}{" "}
                        <p
                          className="ml-2"
                          style={{ color: "gray", fontSize: "12px" }}
                        >
                          {description}
                        </p>
                      </div>
                    </SelectItem>
                  )
                )}
            </Select2>
          </Form.Item>
          <Form.Item label="Team ID" name="team_id">
            <Select placeholder="Select Team ID" style={{ width: "100%" }}>
              {teams ? (
                teams.map((team: any) => (
                  <Option key={team.team_id} value={team.team_id}>
                    {team.team_alias}
                  </Option>
                ))
              ) : (
                <Option key="default" value={null}>
                  Default Team
                </Option>
              )}
            </Select>
          </Form.Item>

          <Form.Item label="Metadata" name="metadata">
            <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
          </Form.Item>
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button htmlType="submit">Create User</Button>
          </div>
        </Form>
      </Modal>
      {apiuser && (
        <OnboardingModal
          isInvitationLinkModalVisible={isInvitationLinkModalVisible}
          setIsInvitationLinkModalVisible={setIsInvitationLinkModalVisible}
          baseUrl={baseUrl}
          invitationLinkData={invitationLinkData}
        />
      )}
    </div>
  );
};

export default Createuser;
