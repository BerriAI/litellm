import React from "react";
import { Form, Input, InputNumber, Select } from "antd";
import { Button } from "@tremor/react";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { all_admin_roles } from "../utils/roles";
interface UserEditViewProps {
  userData: any;
  onCancel: () => void;
  onSubmit: (values: any) => void;
  teams: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  userModels: string[];
}

export function UserEditView({
  userData,
  onCancel,
  onSubmit,
  teams,
  accessToken,
  userID,
  userRole,
  userModels,
}: UserEditViewProps) {
  const [form] = Form.useForm();

  // Set initial form values
  React.useEffect(() => {
    form.setFieldsValue({
      user_id: userData.user_id,
      user_email: userData.user_info?.user_email,
      user_role: userData.user_info?.user_role,
      models: userData.user_info?.models || [],
      max_budget: userData.user_info?.max_budget,
      metadata: userData.user_info?.metadata ? JSON.stringify(userData.user_info.metadata, null, 2) : undefined,
    });
  }, [userData, form]);

  const handleSubmit = (values: any) => {
    // Convert metadata back to an object if it exists and is a string
    if (values.metadata && typeof values.metadata === "string") {
      try {
        values.metadata = JSON.parse(values.metadata);
      } catch (error) {
        console.error("Error parsing metadata JSON:", error);
        return;
      }
    }

    onSubmit(values);
  };

  return (
    <Form
      form={form}
      onFinish={handleSubmit}
      layout="vertical"
    >
      <Form.Item
        label="User ID"
        name="user_id"
      >
        <Input disabled />
      </Form.Item>

      <Form.Item
        label="Email"
        name="user_email"
      >
        <Input />
      </Form.Item>

      <Form.Item
        label="Role"
        name="user_role"
      >
        <Input />
      </Form.Item>

      <Form.Item
        label="Models"
        name="models"
      >
        <Select
          mode="multiple"
          placeholder="Select models"
          style={{ width: "100%" }}
          disabled={!all_admin_roles.includes(userData.user_info?.user_role || "")}
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

      <Form.Item
        label="Max Budget (USD)"
        name="max_budget"
      >
        <InputNumber
          step={0.01}
          precision={2}
          style={{ width: "100%" }}
        />
      </Form.Item>

      <Form.Item
        label="Metadata"
        name="metadata"
      >
        <Input.TextArea
          rows={4}
          placeholder="Enter metadata as JSON"
        />
      </Form.Item>

      <div className="flex justify-end space-x-2">
        <Button variant="secondary" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">
          Save Changes
        </Button>
      </div>
    </Form>
  );
} 