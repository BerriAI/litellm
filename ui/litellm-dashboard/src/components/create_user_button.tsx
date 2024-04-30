import React, { useState, useEffect } from "react";
import { Button, Modal, Form, Input, message, Select, InputNumber } from "antd";
import { Button as Button2, Text, TextInput } from "@tremor/react";
import { userCreateCall, modelAvailableCall } from "./networking";
const { Option } = Select;

interface CreateuserProps {
  userID: string;
  accessToken: string;
  teams: any[] | null;
}

const Createuser: React.FC<CreateuserProps> = ({ userID, accessToken, teams }) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiuser, setApiuser] = useState<string | null>(null);
  const [userModels, setUserModels] = useState<string[]>([]);

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
      message.success("API user Created");
      form.resetFields();
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      console.error("Error creating the user:", error);
    }
  };

  return (
    <div>
      <Button2 className="mx-auto" onClick={() => setIsModalVisible(true)}>
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
        <Text className="mb-1">Invite a user to login to the Admin UI and create Keys</Text>
        <Text className="mb-6"><b>Note: SSO Setup Required for this</b></Text>
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
          <Form.Item label="Team ID" name="team_id">
          <Select
              placeholder="Select Team ID"
              style={{ width: "100%" }}
            >
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
        <Modal
          title="User Created Successfully"
          visible={isModalVisible}
          onOk={handleOk}
          onCancel={handleCancel}
          footer={null}
        >
          <p>
            User has been created to access your proxy. Please Ask them to Log In.
          </p>
          <br></br>

          <p><b>Note: This Feature is only supported through SSO on the Admin UI</b></p>
          
        </Modal>
      )}
    </div>
  );
};

export default Createuser;
