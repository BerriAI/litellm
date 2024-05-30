import { useEffect, useState } from 'react';
import {
  Dialog,
  DialogPanel,
  TextInput,
  Button,
  Select,
  SelectItem,
  Text,
  Title,
  Subtitle,
} from '@tremor/react';

import {
    Button as Button2,
    Modal,
    Form,
    Input,
    Select as Select2,
    InputNumber,
    message,
  } from "antd";

interface EditUserModalProps {
  visible: boolean;
  onCancel: () => void;
  user: any;
  onSubmit: (data: any) => void;
}

const EditUserModal: React.FC<EditUserModalProps> = ({ visible, onCancel, user, onSubmit }) => {
  const [editedUser, setEditedUser] = useState(user);
  const [form] = Form.useForm();

  useEffect(() => {
    form.resetFields();
  }, [user]);

  const handleChange = (e) => {
    setEditedUser({ ...editedUser, [e.target.name]: e.target.value });
  };

  const handleCancel = async () => {
    form.resetFields();
    onCancel();
  };

  const handleEditSubmit = async (formValues: Record<string, any>) => {
    // Call API to update team with teamId and values
    form.resetFields();

    onSubmit(formValues);
    onCancel();
  };



  if (!user) {
    return null;
  }

  return (

    <Modal 
    visible={visible} 
    onCancel={handleCancel} 
    footer={null}
    title={"Edit User " + user.user_id}
    width={1000}
    >
        <Form
          form={form}
          onFinish={handleEditSubmit}
          initialValues={user} // Pass initial values here
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            <Form.Item 
            className="mt-8" 
            label="User Email" 
            tooltip="Email of the User"
            name="user_email">
              <TextInput />
            </Form.Item>

            <Form.Item
              label="user_id"
              name="user_id"
              hidden={true}
            >
              <TextInput />
            </Form.Item>

            <Form.Item
              label="User Role"
              name="user_role"
            >
            <Select2>
                <Select2.Option value="proxy_admin">Proxy Admin (Can create, edit, delete keys, teams)</Select2.Option>
                <Select2.Option value="proxy_admin_viewer">Proxy Viewer (Can just view spend, cannot created keys, teams)</Select2.Option>
            </Select2>

            </Form.Item>

            <Form.Item
              label="Spend (USD)"
              name="spend"
              tooltip="(float) - Spend of all LLM calls completed by this user"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>

            <Form.Item
              label="User Budget (USD)"
              name="max_budget"
              tooltip="(float) - Maximum budget of this user"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>

            <div style={{ textAlign: "right", marginTop: "10px" }}>
                <Button2 htmlType="submit">Save</Button2>
            </div>

        </>

      </Form>

    
    </Modal>
  );
};

export default EditUserModal;