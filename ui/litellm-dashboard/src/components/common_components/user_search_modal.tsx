import { useState, useCallback } from "react";
import { Modal, Form, Button, Select, Tooltip } from "antd";
import { UserAddOutlined } from "@ant-design/icons";
import debounce from "lodash/debounce";
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

interface Role {
  label: string;
  value: string;
  description: string;
}

interface FormValues {
  user_email: string;
  user_id: string;
  role: string;
}

interface UserSearchModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSubmit: (values: FormValues) => void | Promise<void>;
  accessToken: string | null;
  title?: string;
  roles?: Role[];
  defaultRole?: string;
  teamId?: string;
}

const UserSearchModal: React.FC<UserSearchModalProps> = ({
  isVisible,
  onCancel,
  onSubmit,
  accessToken,
  title = "Add Team Member",
  roles = [
    {
      label: "admin",
      value: "admin",
      description: "Admin role. Can create team keys, add members, and manage settings.",
    },
    { label: "user", value: "user", description: "User role. Can view team info, but not manage it." },
  ],
  defaultRole = "user",
  teamId,
}) => {
  const [form] = Form.useForm<FormValues>();
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedField, setSelectedField] = useState<"user_email" | "user_id">("user_email");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Build a synthetic "use what I typed" option so the literal value the
  // caller entered is always selectable. This is what lets team admins (who
  // can't call /user/filter/ui because they lack the proxy-wide list scope)
  // still add a member by typing their email or user_id directly — without
  // this, Antd's Select silently drops the typed text on submit because the
  // value is not part of `options` when `filterOption={false}`.
  const buildFreeTextOption = (
    text: string,
    fieldName: "user_email" | "user_id",
  ): UserOption => {
    const labelPrefix = fieldName === "user_email" ? "Use email" : "Use user ID";
    return {
      label: `${labelPrefix} "${text}"`,
      value: text,
      user:
        fieldName === "user_email"
          ? { user_id: "", user_email: text }
          : { user_id: text, user_email: "" },
    };
  };

  const fetchUsers = async (searchText: string, fieldName: "user_email" | "user_id"): Promise<void> => {
    const trimmed = searchText.trim();
    if (!trimmed) {
      setUserOptions([]);
      return;
    }

    setLoading(true);
    const freeTextOption = buildFreeTextOption(trimmed, fieldName);

    try {
      if (accessToken == null) {
        setUserOptions([freeTextOption]);
        return;
      }
      const params = new URLSearchParams();
      params.append(fieldName, trimmed);
      if (teamId) {
        params.append("team_id", teamId);
      }
      const response = await userFilterUICall(accessToken, params);

      const data: User[] = Array.isArray(response) ? response : [];
      const apiOptions: UserOption[] = data.map((user) => ({
        label: fieldName === "user_email" ? `${user.user_email}` : `${user.user_id}`,
        value: fieldName === "user_email" ? user.user_email : user.user_id,
        user,
      }));

      // If a real result already matches the typed value exactly, omit the
      // synthetic option so the dropdown isn't visually duplicated.
      const apiHasExactMatch = apiOptions.some((opt) => opt.value === trimmed);
      setUserOptions(apiHasExactMatch ? apiOptions : [...apiOptions, freeTextOption]);
    } catch (error) {
      console.error("Error fetching users:", error);
      // Even when /user/filter/ui is unauthorized (typical for team admins),
      // surface the typed value as a selectable option so submit still works.
      setUserOptions([freeTextOption]);
    } finally {
      setLoading(false);
    }
  };

  const debouncedSearch = useCallback(
    debounce((text: string, fieldName: "user_email" | "user_id") => fetchUsers(text, fieldName), 300),
    [],
  );

  const handleSearch = (value: string, fieldName: "user_email" | "user_id"): void => {
    setSelectedField(fieldName);
    debouncedSearch(value, fieldName);
  };

  const handleSelect = (_value: string, option: UserOption): void => {
    const selectedUser = option.user;
    form.setFieldsValue({
      user_email: selectedUser.user_email,
      user_id: selectedUser.user_id,
      role: form.getFieldValue("role"), // Preserve current role selection
    });
  };

  const handleSubmit = async (values: FormValues): Promise<void> => {
    setIsSubmitting(true);
    try {
      await onSubmit(values);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = (): void => {
    form.resetFields();
    setUserOptions([]);
    onCancel();
  };

  return (
    <Modal title={title} open={isVisible} onCancel={handleClose} footer={null} width={800} maskClosable={!isSubmitting}>
      <Form<FormValues>
        form={form}
        onFinish={handleSubmit}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        initialValues={{
          role: defaultRole,
        }}
      >
        <Form.Item label="Email" name="user_email" className="mb-4">
          <Select
            showSearch
            className="w-full"
            placeholder="Search by email"
            filterOption={false}
            onSearch={(value) => handleSearch(value, "user_email")}
            onSelect={(value, option) => handleSelect(value, option as UserOption)}
            options={selectedField === "user_email" ? userOptions : []}
            loading={loading}
            allowClear
            data-testid="member-email-search"
          />
        </Form.Item>

        <div className="text-center mb-4">OR</div>

        <Form.Item label="User ID" name="user_id" className="mb-4">
          <Select
            showSearch
            className="w-full"
            placeholder="Search by user ID"
            filterOption={false}
            onSearch={(value) => handleSearch(value, "user_id")}
            onSelect={(value, option) => handleSelect(value, option as UserOption)}
            options={selectedField === "user_id" ? userOptions : []}
            loading={loading}
            allowClear
            data-testid="member-userid-search"
          />
        </Form.Item>

        <Form.Item label="Member Role" name="role" className="mb-4">
          <Select defaultValue={defaultRole}>
            {roles.map((role) => (
              <Select.Option key={role.value} value={role.value}>
                <Tooltip title={role.description}>
                  <span className="font-medium">{role.label}</span>
                  <span className="ml-2 text-gray-500 text-sm">- {role.description}</span>
                </Tooltip>
              </Select.Option>
            ))}
          </Select>
        </Form.Item>

        <div className="text-right mt-4">
          <Button type="primary" htmlType="submit" icon={<UserAddOutlined />} loading={isSubmitting}>
            {isSubmitting ? "Adding..." : "Add Member"}
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default UserSearchModal;
