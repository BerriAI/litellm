/**
 * This component is used to add an admin to an organization.
 */
import React, { FC } from "react";
import { Button, Col, Text } from "@tremor/react";
import { Button as Button2, Select as Select2, Modal, Form, Input } from "antd";
import { Organization } from "@/components/organization/types";
interface AddOrgAdminProps {
  userRole: string;
  userID: string;
  selectedOrganization?: Organization;
  onMemberAdd?: (formValues: Record<string, any>) => void;
}

const is_org_admin = (organization: any, userID: string) => {
  for (let i = 0; i < organization.members_with_roles.length; i++) {
    let member = organization.members_with_roles[i];
    if (member.user_id == userID && member.role == "admin") {
      return true;
    }
  }
  return false;
};

const AddOrgAdmin: FC<AddOrgAdminProps> = ({ userRole, userID, selectedOrganization, onMemberAdd }) => {
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = React.useState(false);
  const [form] = Form.useForm();

  const handleMemberCancel = () => {
    form.resetFields();
    setIsAddMemberModalVisible(false);
  };

  const handleMemberOk = () => {
    form.submit();
  };

  return (
    <Col numColSpan={1}>
      {userRole === "Admin" || (selectedOrganization && is_org_admin(selectedOrganization, userID)) ? (
        <Button className="mx-auto mb-5" onClick={() => setIsAddMemberModalVisible(true)}>
          + Add member
        </Button>
      ) : null}

      <Modal
        title="Add member"
        visible={isAddMemberModalVisible}
        width={800}
        footer={null}
        onOk={handleMemberOk}
        onCancel={handleMemberCancel}
      >
        <Text className="mb-2">User must exist in proxy. Get User ID from &apos;Users&apos; tab.</Text>
        <Form
          form={form}
          onFinish={onMemberAdd}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
          initialValues={{
            role: "internal_user",
          }}
        >
          <Form.Item label="Email" name="user_email" className="mb-4">
            <Input name="user_email" className="px-3 py-2 border rounded-md w-full" />
          </Form.Item>

          <div className="text-center mb-4">OR</div>

          <Form.Item label="User ID" name="user_id" className="mb-4">
            <Input name="user_id" className="px-3 py-2 border rounded-md w-full" />
          </Form.Item>

          <Form.Item label="Member Role" name="role" className="mb-4">
            <Select2 defaultValue="user">
              <Select2.Option value="org_admin">
                <div className="flex">
                  Org Admin{" "}
                  <p className="ml-2" style={{ color: "gray", fontSize: "12px" }}>
                    Can add and remove members, and change their roles.
                  </p>
                </div>
              </Select2.Option>
              <Select2.Option value="internal_user">
                <div className="flex">
                  Internal User{" "}
                  <p className="ml-2" style={{ color: "gray", fontSize: "12px" }}>
                    Can view/create keys for themselves within organization.
                  </p>
                </div>
              </Select2.Option>
              <Select2.Option value="internal_user_viewer">
                <div className="flex">
                  Internal User Viewer{" "}
                  <p className="ml-2" style={{ color: "gray", fontSize: "12px" }}>
                    Can only view their keys within organization.
                  </p>
                </div>
              </Select2.Option>
            </Select2>
          </Form.Item>

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Add member</Button2>
          </div>
        </Form>
      </Modal>
    </Col>
  );
};

export default AddOrgAdmin;
