import React, { useEffect } from "react";
import { Modal, Form, Button as AntButton } from "antd";
import { Select, SelectItem, TextInput } from "@tremor/react";
import { Text } from "@tremor/react";
import NumericalInput from "../shared/numerical_input";

interface BaseMember {
  user_email?: string;
  user_id?: string;
  role: string;
}

interface ModalConfig {
  title: string;
  roleOptions: Array<{
    label: string;
    value: string;
  }>;
  defaultRole?: string;
  showEmail?: boolean;
  showUserId?: boolean;
  additionalFields?: Array<{
    name: string;
    label: string | React.ReactNode;
    type: "input" | "select" | "numerical";
    options?: Array<{ label: string; value: string }>;
    rules?: any[];
    step?: number;
    min?: number;
    placeholder?: string;
  }>;
}

interface MemberModalProps<T extends BaseMember> {
  visible: boolean;
  onCancel: () => void;
  onSubmit: (data: T) => void;
  initialData?: T | null;
  mode: "add" | "edit";
  config: ModalConfig;
}

const MemberModal = <T extends BaseMember>({
  visible,
  onCancel,
  onSubmit,
  initialData,
  mode,
  config,
}: MemberModalProps<T>) => {
  const [form] = Form.useForm();

  console.log("Initial Data:", initialData);

  // Reset form and set initial values when modal becomes visible or initialData changes
  useEffect(() => {
    if (visible) {
      if (mode === "edit" && initialData) {
        // For edit mode, use the initialData values
        const formValues = {
          ...initialData,
          // Ensure role is set correctly for editing
          role: initialData.role || config.defaultRole,
          // Keep numeric values as numbers for NumericalInput components
          max_budget_in_team: (initialData as any).max_budget_in_team || null,
          tpm_limit: (initialData as any).tpm_limit || null,
          rpm_limit: (initialData as any).rpm_limit || null,
        };
        console.log("Setting form values:", formValues);
        form.setFieldsValue(formValues);
      } else {
        // For add mode, reset to defaults
        form.resetFields();
        form.setFieldsValue({
          role: config.defaultRole || config.roleOptions[0]?.value,
        });
      }
    }
  }, [visible, initialData, mode, form, config.defaultRole, config.roleOptions]);

  const handleSubmit = async (values: any) => {
    try {
      // Trim string values and clean up form data
      const formData = Object.entries(values).reduce((acc, [key, value]) => {
        if (typeof value === "string") {
          const trimmedValue = value.trim();
          // For empty strings on optional numeric fields, set to null
          if (trimmedValue === "" && (key === "max_budget_in_team" || key === "tpm_limit" || key === "rpm_limit")) {
            return { ...acc, [key]: null };
          }
          return { ...acc, [key]: trimmedValue };
        }
        // For numeric values from NumericalInput, use as-is (already numbers)
        return { ...acc, [key]: value };
      }, {}) as T;

      console.log("Submitting form data:", formData);
      onSubmit(formData);
      form.resetFields();
      // NotificationsManager.success(`Successfully ${mode === 'add' ? 'added' : 'updated'} member`);
    } catch (error) {
      // NotificationManager.fromBackend('Failed to submit form');
      console.error("Form submission error:", error);
    }
  };

  // Helper function to get role label from value
  const getRoleLabel = (value: string) => {
    return config.roleOptions.find((option) => option.value === value)?.label || value;
  };

  const renderField = (field: {
    name: string;
    label: string | React.ReactNode;
    type: "input" | "select" | "numerical";
    options?: Array<{ label: string; value: string }>;
    rules?: any[];
    step?: number;
    min?: number;
    placeholder?: string;
  }) => {
    switch (field.type) {
      case "input":
        return <TextInput placeholder={field.placeholder} />;
      case "numerical":
        return (
          <NumericalInput
            step={field.step || 1}
            min={field.min || 0}
            style={{ width: "100%" }}
            placeholder={field.placeholder || "Enter a numerical value"}
          />
        );
      case "select":
        return (
          <Select>
            {field.options?.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </Select>
        );
      default:
        return null;
    }
  };

  return (
    <Modal
      title={config.title || (mode === "add" ? "Add Member" : "Edit Member")}
      open={visible}
      width={1000}
      footer={null}
      onCancel={onCancel}
    >
      <Form form={form} onFinish={handleSubmit} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
        {config.showEmail && (
          <Form.Item
            label="Email"
            name="user_email"
            className="mb-4"
            rules={[{ type: "email", message: "Please enter a valid email!" }]}
          >
            <TextInput placeholder="user@example.com" />
          </Form.Item>
        )}

        {config.showEmail && config.showUserId && (
          <div className="text-center mb-4">
            <Text>OR</Text>
          </div>
        )}

        {config.showUserId && (
          <Form.Item label="User ID" name="user_id" className="mb-4">
            <TextInput placeholder="user_123" />
          </Form.Item>
        )}

        <Form.Item
          label={
            <div className="flex items-center gap-2">
              <span>Role</span>
              {mode === "edit" && initialData && (
                <span className="text-gray-500 text-sm">(Current: {getRoleLabel(initialData.role)})</span>
              )}
            </div>
          }
          name="role"
          className="mb-4"
          rules={[{ required: true, message: "Please select a role!" }]}
        >
          <Select>
            {mode === "edit" && initialData
              ? [
                  // Current role first
                  ...config.roleOptions.filter((option) => option.value === initialData.role),
                  // Then all other roles
                  ...config.roleOptions.filter((option) => option.value !== initialData.role),
                ].map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))
              : config.roleOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
          </Select>
        </Form.Item>

        {config.additionalFields?.map((field) => (
          <Form.Item key={field.name} label={field.label} name={field.name} className="mb-4" rules={field.rules}>
            {renderField(field)}
          </Form.Item>
        ))}

        <div className="text-right mt-6">
          <AntButton onClick={onCancel} className="mr-2">
            Cancel
          </AntButton>
          <AntButton type="default" htmlType="submit">
            {mode === "add" ? "Add Member" : "Save Changes"}
          </AntButton>
        </div>
      </Form>
    </Modal>
  );
};

export default MemberModal;
