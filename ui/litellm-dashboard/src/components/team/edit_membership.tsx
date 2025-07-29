import React, { useState, useEffect } from 'react';
import { Modal, Form, Input, Select as AntSelect, Button as AntButton, message } from 'antd';
import { Select, SelectItem } from "@tremor/react";
import { Card, Text } from "@tremor/react";

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
    label: string;
    type: 'input' | 'select';
    options?: Array<{ label: string; value: string }>;
    rules?: any[];
  }>;
}

interface MemberModalProps<T extends BaseMember> {
  visible: boolean;
  onCancel: () => void;
  onSubmit: (data: T) => void;
  initialData?: T | null;
  mode: 'add' | 'edit';
  config: ModalConfig;
}

const MemberModal = <T extends BaseMember>({
  visible,
  onCancel,
  onSubmit,
  initialData,
  mode,
  config
}: MemberModalProps<T>) => {
  const [form] = Form.useForm();

  console.log("Initial Data:", initialData);

  // Reset form and set initial values when modal becomes visible or initialData changes
  useEffect(() => {
    if (visible) {
      if (mode === 'edit' && initialData) {
        // For edit mode, use the initialData values
        form.setFieldsValue({
          ...initialData,
          // Ensure role is set correctly for editing
          role: initialData.role || config.defaultRole
        });
      } else {
        // For add mode, reset to defaults
        form.resetFields();
        form.setFieldsValue({
          role: config.defaultRole || config.roleOptions[0]?.value
        });
      }
    }
  }, [visible, initialData, mode, form, config.defaultRole, config.roleOptions]);

  const handleSubmit = async (values: any) => {
    try {
      // Trim string values
      const formData = Object.entries(values).reduce((acc, [key, value]) => ({
        ...acc,
        [key]: typeof value === 'string' ? value.trim() : value
      }), {}) as T;
      
      onSubmit(formData);
      form.resetFields();
      // message.success(`Successfully ${mode === 'add' ? 'added' : 'updated'} member`);
    } catch (error) {
      // message.error('Failed to submit form');
      console.error('Form submission error:', error);
    }
  };

  // Helper function to get role label from value
  const getRoleLabel = (value: string) => {
    return config.roleOptions.find(option => option.value === value)?.label || value;
  };

  const renderField = (field: {
    name: string;
    label: string;
    type: 'input' | 'select';
    options?: Array<{ label: string; value: string }>;
    rules?: any[];
  }) => {
    switch (field.type) {
      case 'input':
        return (
          <Input
            className="px-3 py-2 border rounded-md w-full"
            onChange={(e) => {
              e.target.value = e.target.value.trim();
            }}
          />
        );
      case 'select':
        return (
          <AntSelect>
            {field.options?.map(option => (
              <AntSelect.Option key={option.value} value={option.value}>
                {option.label}
              </AntSelect.Option>
            ))}
          </AntSelect>
        );
      default:
        return null;
    }
  };

  return (
    <Modal
      title={config.title || (mode === 'add' ? "Add Member" : "Edit Member")}
      open={visible}
      width={800}
      footer={null}
      onCancel={onCancel}
    >
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
              { type: 'email', message: 'Please enter a valid email!' }
            ]}
          >
            <Input
              className="px-3 py-2 border rounded-md w-full"
              placeholder="user@example.com"
              onChange={(e) => {
                e.target.value = e.target.value.trim();
              }}
            />
          </Form.Item>
        )}

        {config.showEmail && config.showUserId && (
          <div className="text-center mb-4">
            <Text>OR</Text>
          </div>
        )}

        {config.showUserId && (
          <Form.Item 
            label="User ID" 
            name="user_id"
            className="mb-4"
          >
            <Input
              className="px-3 py-2 border rounded-md w-full"
              placeholder="user_123"
              onChange={(e) => {
                e.target.value = e.target.value.trim();
              }}
            />
          </Form.Item>
        )}

        <Form.Item 
          label={
            <div className="flex items-center gap-2">
              <span>Role</span>
              {mode === 'edit' && initialData && (
                <span className="text-gray-500 text-sm">
                  (Current: {getRoleLabel(initialData.role)})
                </span>
              )}
            </div>
          }
          name="role"
          className="mb-4"
          rules={[
            { required: true, message: 'Please select a role!' }
          ]}
        >
          <AntSelect>
            {mode === 'edit' && initialData
              ? [
                  // Current role first
                  ...config.roleOptions.filter(option => option.value === initialData.role),
                  // Then all other roles
                  ...config.roleOptions.filter(option => option.value !== initialData.role)
                ].map(option => (
                  <AntSelect.Option key={option.value} value={option.value}>
                    {option.label}
                  </AntSelect.Option>
                ))
              : config.roleOptions.map(option => (
                  <AntSelect.Option key={option.value} value={option.value}>
                    {option.label}
                  </AntSelect.Option>
                ))}
          </AntSelect>
        </Form.Item>

        {config.additionalFields?.map(field => (
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
          <AntButton 
            onClick={onCancel} 
            className="mr-2"
          >
            Cancel
          </AntButton>
          <AntButton 
            type="default" 
            htmlType="submit"
          >
            {mode === 'add' ? 'Add Member' : 'Save Changes'}
          </AntButton>
        </div>
      </Form>
    </Modal>
  );
};

export default MemberModal;