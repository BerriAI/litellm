import { useState, useCallback } from 'react';
import { Modal, Form, Button, Select } from 'antd';
import debounce from 'lodash/debounce';
import { userFilterUICall } from "@/components/networking";

interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

interface UserOption {
  label: string;
  value: string;
  user: User;
}

interface FormValues {
  user_email: string;
  user_id: string;
  role: 'admin' | 'user';
}

interface UserSearchModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSubmit: (values: FormValues) => void;
  accessToken: string | null;
}

const UserSearchModal: React.FC<UserSearchModalProps> = ({ 
  isVisible, 
  onCancel, 
  onSubmit,
  accessToken
}) => {
  const [form] = Form.useForm<FormValues>();
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedField, setSelectedField] = useState<'user_email' | 'user_id'>('user_email');

  const fetchUsers = async (searchText: string, fieldName: 'user_email' | 'user_id'): Promise<void> => {
    if (!searchText) {
      setUserOptions([]);
      return;
    }

    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.append(fieldName, searchText);
      if (accessToken == null) {
        return;
      }
      const response = await userFilterUICall(accessToken, params);
      
        const data: User[] = response
        const options: UserOption[] = data.map(user => ({
            label: fieldName === 'user_email' 
                ? `${user.user_email}`
                : `${user.user_id}`,
            value: fieldName === 'user_email' ? user.user_email : user.user_id,
            user
         }));
        setUserOptions(options);
    } catch (error) {
      console.error('Error fetching users:', error);
    } finally {
      setLoading(false);
    }
  };

  const debouncedSearch = useCallback(
    debounce((text: string, fieldName: 'user_email' | 'user_id') => fetchUsers(text, fieldName), 300),
    []
  );

  const handleSearch = (value: string, fieldName: 'user_email' | 'user_id'): void => {
    setSelectedField(fieldName);
    debouncedSearch(value, fieldName);
  };

  const handleSelect = (_value: string, option: UserOption): void => {
    const selectedUser = option.user;
    form.setFieldsValue({
      user_email: selectedUser.user_email,
      user_id: selectedUser.user_id,
      role: form.getFieldValue('role') // Preserve current role selection
    });
  };

  const handleClose = (): void => {
    form.resetFields();
    setUserOptions([]);
    onCancel();
  };

  return (
    <Modal
      title="Add Team Member" 
      open={isVisible}
      onCancel={handleClose}
      footer={null}
      width={800}
    >
      <Form<FormValues>
        form={form}
        onFinish={onSubmit}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        initialValues={{
          role: "user",
        }}
      >
        <Form.Item
          label="Email"
          name="user_email"
          className="mb-4"
        >
          <Select
            showSearch
            className="w-full" 
            placeholder="Search by email"
            filterOption={false}
            onSearch={(value) => handleSearch(value, 'user_email')}
            onSelect={(value, option) => handleSelect(value, option as UserOption)}
            options={selectedField === 'user_email' ? userOptions : []}
            loading={loading}
            allowClear
          />
        </Form.Item>

        <div className="text-center mb-4">OR</div>
        
        <Form.Item
          label="User ID"
          name="user_id" 
          className="mb-4"
        >
          <Select
            showSearch
            className="w-full"
            placeholder="Search by user ID" 
            filterOption={false}
            onSearch={(value) => handleSearch(value, 'user_id')}
            onSelect={(value, option) => handleSelect(value, option as UserOption)}
            options={selectedField === 'user_id' ? userOptions : []}
            loading={loading}
            allowClear
          />
        </Form.Item>

        <Form.Item
          label="Member Role"
          name="role"
          className="mb-4"
        >
          <Select defaultValue="user">
            <Select.Option value="admin">admin</Select.Option>
            <Select.Option value="user">user</Select.Option>
          </Select>
        </Form.Item>

        <div className="text-right mt-4">
          <Button type="primary" htmlType="submit">
            Add Member
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default UserSearchModal;