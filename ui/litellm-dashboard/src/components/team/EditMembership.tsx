import { Form, Select } from "antd";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import React, { useEffect, useState } from "react";
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
    type: "input" | "select" | "numerical" | "multi-select";
    options?: Array<{ label: string; value: string }>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Reset form and set initial values when modal becomes visible or initialData changes
  useEffect(() => {
    if (visible) {
      if (mode === "edit" && initialData) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const d = initialData as any;
        const formValues = {
          ...initialData,
          role: initialData.role || config.defaultRole,
          max_budget_in_team: d.max_budget_in_team || null,
          tpm_limit: d.tpm_limit || null,
          rpm_limit: d.rpm_limit || null,
          allowed_models: d.allowed_models || [],
        };
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleSubmit = async (values: any) => {
    try {
      setIsSubmitting(true);
      const formData = Object.entries(values).reduce((acc, [key, value]) => {
        if (typeof value === "string") {
          const trimmedValue = value.trim();
          if (
            trimmedValue === "" &&
            (key === "max_budget_in_team" ||
              key === "tpm_limit" ||
              key === "rpm_limit")
          ) {
            return { ...acc, [key]: null };
          }
          return { ...acc, [key]: trimmedValue };
        }
        return { ...acc, [key]: value };
      }, {}) as T;

      await Promise.resolve(onSubmit(formData));
      form.resetFields();
    } catch (error) {
      console.error("Form submission error:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Helper function to get role label from value
  const getRoleLabel = (value: string) => {
    return config.roleOptions.find((option) => option.value === value)?.label || value;
  };

  const renderField = (field: {
    name: string;
    label: string | React.ReactNode;
    type: "input" | "select" | "numerical" | "multi-select";
    options?: Array<{ label: string; value: string }>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    rules?: any[];
    step?: number;
    min?: number;
    placeholder?: string;
  }) => {
    switch (field.type) {
      case "input":
        return <Input placeholder={field.placeholder} />;
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
              <Select.Option key={option.value} value={option.value}>
                {option.label}
              </Select.Option>
            ))}
          </Select>
        );
      case "multi-select":
        return (
          <Select
            mode="multiple"
            placeholder={field.placeholder || "Select options"}
            options={field.options}
            allowClear
          />
        );
      default:
        return null;
    }
  };

  return (
    <Dialog open={visible} onOpenChange={(o) => (!o ? onCancel() : undefined)}>
      <DialogContent className="max-w-[1000px]">
        <DialogHeader>
          <DialogTitle>
            {config.title || (mode === "add" ? "Add Member" : "Edit Member")}
          </DialogTitle>
        </DialogHeader>
        <Form
          form={form}
          onFinish={handleSubmit}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          {config.showEmail && (
            <Form.Item
              label="Email"
              name="user_email"
              className="mb-4"
              rules={[
                { type: "email", message: "Please enter a valid email!" },
              ]}
            >
              <Input placeholder="user@example.com" />
            </Form.Item>
          )}

          {config.showEmail && config.showUserId && (
            <div className="text-center mb-4">
              <span>OR</span>
            </div>
          )}

          {config.showUserId && (
            <Form.Item label="User ID" name="user_id" className="mb-4">
              <Input placeholder="user_123" />
            </Form.Item>
          )}

          <Form.Item
            label={
              <div className="flex items-center gap-2">
                <span>Role</span>
                {mode === "edit" && initialData && (
                  <span className="text-muted-foreground text-sm">
                    (Current: {getRoleLabel(initialData.role)})
                  </span>
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
                    ...config.roleOptions.filter(
                      (option) => option.value === initialData.role,
                    ),
                    ...config.roleOptions.filter(
                      (option) => option.value !== initialData.role,
                    ),
                  ].map((option) => (
                    <Select.Option key={option.value} value={option.value}>
                      {option.label}
                    </Select.Option>
                  ))
                : config.roleOptions.map((option) => (
                    <Select.Option key={option.value} value={option.value}>
                      {option.label}
                    </Select.Option>
                  ))}
            </Select>
          </Form.Item>

          {config.additionalFields?.map((field) => (
            <Form.Item
              key={field.name}
              label={field.label}
              name={field.name}
              className="mb-4"
              rules={field.rules}
            >
              {renderField(field)}
            </Form.Item>
          ))}

          <div className="text-right mt-6">
            <Button
              type="button"
              variant="outline"
              onClick={onCancel}
              className="mr-2"
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {mode === "add"
                ? isSubmitting
                  ? "Adding..."
                  : "Add Member"
                : isSubmitting
                  ? "Saving..."
                  : "Save Changes"}
            </Button>
          </div>
        </Form>
      </DialogContent>
    </Dialog>
  );
};

export default MemberModal;
