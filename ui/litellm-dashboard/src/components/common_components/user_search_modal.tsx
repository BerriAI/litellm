import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Form, Select } from "antd";
import { UserPlus } from "lucide-react";
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
    debounce(
      (text: string, fieldName: "user_email" | "user_id") =>
        fetchUsers(text, fieldName),
      300,
    ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o && !isSubmitting ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
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
              onSelect={(value, option) =>
                handleSelect(value, option as UserOption)
              }
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
              onSelect={(value, option) =>
                handleSelect(value, option as UserOption)
              }
              options={selectedField === "user_id" ? userOptions : []}
              loading={loading}
              allowClear
            />
          </Form.Item>

          <Form.Item label="Member Role" name="role" className="mb-4">
            <Select defaultValue={defaultRole}>
              {roles.map((role) => (
                <Select.Option key={role.value} value={role.value}>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span>
                          <span className="font-medium">{role.label}</span>
                          <span className="ml-2 text-muted-foreground text-sm">
                            - {role.description}
                          </span>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>{role.description}</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Select.Option>
              ))}
            </Select>
          </Form.Item>

          <div className="text-right mt-4">
            <Button type="submit" disabled={isSubmitting}>
              <UserPlus className="h-4 w-4" />
              {isSubmitting ? "Adding..." : "Add Member"}
            </Button>
          </div>
        </Form>
      </DialogContent>
    </Dialog>
  );
};

export default UserSearchModal;
