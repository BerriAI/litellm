import React from "react";
import { Select, TextInput } from "@tremor/react";
import { Form, Select as AntSelect } from "antd";
import TeamDropdown from "./team_dropdown";
import { getPossibleUserRoles } from "../networking";
import TextArea from "antd/es/input/TextArea";

interface UserFormProps {
  form: any;
  teams: any[] | null;
  possibleUIRoles: null | Record<string, Record<string, string>>;
  setPossibleUIRoles?: (roles: any) => void;
  accessToken?: string;
}

const UserForm: React.FC<UserFormProps> = ({ form, teams, possibleUIRoles, setPossibleUIRoles, accessToken }) => {
  React.useEffect(() => {
    // Fetch roles if they're not available and we have a setter
    if (!possibleUIRoles && setPossibleUIRoles && accessToken) {
      getPossibleUserRoles(accessToken).then((roles) => {
        setPossibleUIRoles(roles);
      });
    }
  }, [possibleUIRoles, setPossibleUIRoles, accessToken]);

  return (
    <>
      <Form.Item label="User Email" name="user_email" rules={[{ required: true, message: "Please input user email" }]}>
        <TextInput placeholder="" />
      </Form.Item>

      <Form.Item label="User Role" name="user_role" rules={[{ required: true, message: "Please select a role" }]}>
        <Select>
          {possibleUIRoles &&
            Object.entries(possibleUIRoles).map(([role, { ui_label, description }]) => (
              <AntSelect.Option key={role} value={role} title={ui_label}>
                <div className="flex">
                  {ui_label}{" "}
                  <p className="ml-2" style={{ color: "gray", fontSize: "12px" }}>
                    {description}
                  </p>
                </div>
              </AntSelect.Option>
            ))}
        </Select>
      </Form.Item>

      <Form.Item label="Team" name="team_id">
        <TeamDropdown teams={teams} />
      </Form.Item>

      <Form.Item label="Metadata" name="metadata">
        <TextArea rows={4} placeholder="Enter metadata as JSON" />
      </Form.Item>
    </>
  );
};

export default UserForm;
