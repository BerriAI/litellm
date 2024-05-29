import { useState } from 'react';
import {
  Dialog,
  DialogPanel,
  TextInput,
  Button,
  Select,
  SelectItem,
  Text,
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

  const handleChange = (e) => {
    setEditedUser({ ...editedUser, [e.target.name]: e.target.value });
  };

  const handleSubmit = () => {
    onSubmit(editedUser);
    onCancel();
  };

  if (!user) {
    return null;
  }

  return (

    <Modal visible={visible} onCancel={onCancel}>

        <Text>
            {JSON.stringify(user)}
        </Text>
        
        <Form
          form={form}
          onFinish={handleSubmit}
          initialValues={user} // Pass initial values here
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            <Form.Item 
            className="mt-8" 
            label="user_email" 
            name="user_email">
              <TextInput />
            </Form.Item>

            <Form.Item
              label="user_id"
              name="user_id"
              tooltip="int (optional) - Tokens limit for this deployment: in tokens per minute (tpm). Find this information on your model/providers website"
            >
              <TextInput />
            </Form.Item>

            <Form.Item
              label="user_role"
              name="user_role"
              tooltip="int (optional) - Tokens limit for this deployment: in tokens per minute (tpm). Find this information on your model/providers website"
            >
              <TextInput />
            </Form.Item>

            <Form.Item
              label="spend"
              name="spend"
              tooltip="int (optional) - Tokens limit for this deployment: in tokens per minute (tpm). Find this information on your model/providers website"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>

            <Form.Item
              label="max_budget"
              name="max_budget"
              tooltip="int (optional) - Tokens limit for this deployment: in tokens per minute (tpm). Find this information on your model/providers website"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>

            <Form.Item
              label="rpm"
              name="rpm"
              tooltip="int (optional) - Rate limit for this deployment: in requests per minute (rpm). Find this information on your model/providers website"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>
        <Button onClick={handleSubmit}>Save</Button>
        </>

      </Form>

    
    </Modal>
  );
};

export default EditUserModal;