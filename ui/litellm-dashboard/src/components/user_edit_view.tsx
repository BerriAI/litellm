import React from "react";
import { Form, InputNumber, Select, Tooltip } from "antd";
import NumericalInput from "./shared/numerical_input";
import { TextInput, Textarea, SelectItem } from "@tremor/react";
import { Button } from "@tremor/react";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { all_admin_roles } from "../utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import BudgetDurationDropdown from "./common_components/budget_duration_dropdown";

interface UserEditViewProps {
  userData: any;
  onCancel: () => void;
  onSubmit: (values: any) => void;
  teams: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  userModels: string[];
  possibleUIRoles: Record<string, Record<string, string>> | null;
  isBulkEdit?: boolean;
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
  possibleUIRoles,
  isBulkEdit = false,
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
      budget_duration: userData.user_info?.budget_duration,
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
      {!isBulkEdit && (
        <Form.Item
          label="User ID"
          name="user_id"
        >
          <TextInput disabled />
        </Form.Item>
      )}

      {!isBulkEdit && (
        <Form.Item
          label="Email"
          name="user_email"
        >
          <TextInput />
        </Form.Item>
      )}

      <Form.Item label={
                  <span>
                    Global Proxy Role{' '}
                    <Tooltip title="This is the role that the user will globally on the proxy. This role is independent of any team/org specific roles.">
                      <InfoCircleOutlined/>
                    </Tooltip>
                  </span>
                } 
              name="user_role">
            <Select>
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
                  ),
                )}
            </Select>
          </Form.Item>

      <Form.Item
        label={
          <span>
            Personal Models{' '}
            <Tooltip title="Select which models this user can access outside of team-scope. Choose 'All Proxy Models' to grant access to all models available on the proxy.">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="models"
      >
        <Select
          mode="multiple"
          placeholder="Select models"
          style={{ width: "100%" }}
          disabled={!all_admin_roles.includes(userRole || "")}
          
        >
          <Select.Option key="all-proxy-models" value="all-proxy-models">
            All Proxy Models
          </Select.Option>
          <Select.Option key="no-default-models" value="no-default-models">
            No Default Models
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
        <NumericalInput
          step={0.01}
          precision={2}
          style={{ width: "100%" }}
        />
      </Form.Item>

      <Form.Item label="Reset Budget" name="budget_duration">
        <BudgetDurationDropdown />
      </Form.Item>

      <Form.Item
        label="Metadata"
        name="metadata"
      >
        <Textarea
          rows={4}
          placeholder="Enter metadata as JSON"
        />
      </Form.Item>

      <div className="flex justify-end space-x-2">
        <Button variant="secondary" type="button" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">
          Save Changes
        </Button>
      </div>
    </Form>
  );
} 