import { useState, useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
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
  title,
  roles,
  defaultRole = "user",
  teamId,
}) => {
  const { t } = useTranslation();
  const [form] = Form.useForm<FormValues>();

  const defaultRoles = useMemo(
    () => [
      {
        label: t("commonComponents.userSearchModal.roleAdminLabel"),
        value: "admin",
        description: t("commonComponents.userSearchModal.roleAdminDescription"),
      },
      {
        label: t("commonComponents.userSearchModal.roleUserLabel"),
        value: "user",
        description: t("commonComponents.userSearchModal.roleUserDescription"),
      },
    ],
    [t],
  );

  const effectiveRoles = roles ?? defaultRoles;
  const effectiveTitle = title ?? t("commonComponents.userSearchModal.title");
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedField, setSelectedField] = useState<"user_email" | "user_id">("user_email");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const fetchUsers = async (searchText: string, fieldName: "user_email" | "user_id"): Promise<void> => {
    if (!searchText) {
      setUserOptions([]);
      return;
    }

    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.append(fieldName, searchText);
      if (teamId) {
        params.append("team_id", teamId);
      }
      if (accessToken == null) {
        return;
      }
      const response = await userFilterUICall(accessToken, params);

      const data: User[] = response;
      const options: UserOption[] = data.map((user) => ({
        label: fieldName === "user_email" ? `${user.user_email}` : `${user.user_id}`,
        value: fieldName === "user_email" ? user.user_email : user.user_id,
        user,
      }));
      setUserOptions(options);
    } catch (error) {
      console.error("Error fetching users:", error);
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
    <Modal
      title={effectiveTitle}
      open={isVisible}
      onCancel={handleClose}
      footer={null}
      width={800}
      maskClosable={!isSubmitting}
    >
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
        <Form.Item label={t("commonComponents.userSearchModal.emailLabel")} name="user_email" className="mb-4">
          <Select
            showSearch
            className="w-full"
            placeholder={t("commonComponents.userSearchModal.searchByEmailPlaceholder")}
            filterOption={false}
            onSearch={(value) => handleSearch(value, "user_email")}
            onSelect={(value, option) => handleSelect(value, option as UserOption)}
            options={selectedField === "user_email" ? userOptions : []}
            loading={loading}
            allowClear
            data-testid="member-email-search"
          />
        </Form.Item>

        <div className="text-center mb-4">{t("commonComponents.userSearchModal.orDivider")}</div>

        <Form.Item label={t("commonComponents.userSearchModal.userIdLabel")} name="user_id" className="mb-4">
          <Select
            showSearch
            className="w-full"
            placeholder={t("commonComponents.userSearchModal.searchByUserIdPlaceholder")}
            filterOption={false}
            onSearch={(value) => handleSearch(value, "user_id")}
            onSelect={(value, option) => handleSelect(value, option as UserOption)}
            options={selectedField === "user_id" ? userOptions : []}
            loading={loading}
            allowClear
          />
        </Form.Item>

        <Form.Item label={t("commonComponents.userSearchModal.memberRoleLabel")} name="role" className="mb-4">
          <Select defaultValue={defaultRole}>
            {effectiveRoles.map((role) => (
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
            {isSubmitting
              ? t("commonComponents.userSearchModal.adding")
              : t("commonComponents.userSearchModal.addMember")}
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default UserSearchModal;
