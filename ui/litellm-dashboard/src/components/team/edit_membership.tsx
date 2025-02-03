import React, { useState, useEffect } from 'react';
import { Modal, Form, Input, Select as AntSelect, Button as AntButton, message } from 'antd';
import { Select, SelectItem } from "@tremor/react";
import { Card, Text } from "@tremor/react";
import { Member } from "@/components/networking";
interface TeamMemberModalProps {
  visible: boolean;
  onCancel: () => void;
  onSubmit: (data: Member) => void;
  initialData?: Member | null;
  mode: 'add' | 'edit';
}

const TeamMemberModal: React.FC<TeamMemberModalProps> = ({
  visible,
  onCancel,
  onSubmit,
  initialData,
  mode
}) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (initialData) {
      form.setFieldsValue({
        user_email: initialData.user_email,
        user_id: initialData.user_id,
        role: initialData.role,
      });
    }
  }, [initialData, form]);
  

  const handleSubmit = async (values: any) => {
    try {
      const formData: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };
      
      onSubmit(formData);
      form.resetFields();
      message.success(`Successfully ${mode === 'add' ? 'added' : 'updated'} team member`);
    } catch (error) {
      message.error('Failed to submit form');
      console.error('Form submission error:', error);
    }
  };



  return (
    <Modal
      title={mode === 'add' ? "Add Team Member" : "Edit Team Member"}
      visible={visible}
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
        initialValues={{
          user_email: initialData?.user_email?.trim() || '',
          user_id: initialData?.user_id?.trim() || '',
          role: initialData?.role || 'user',
        }}
      >
        <Form.Item 
          label="Email" 
          name="user_email"
          className="mb-4"
          rules={[
            { type: 'email', message: 'Please enter a valid email!' }
          ]}
        >
          <Input
            name="user_email"
            className="px-3 py-2 border rounded-md w-full"
            placeholder="user@example.com"
            onChange={(e) => {
                e.target.value = e.target.value.trim();
              }} 
          />
        </Form.Item>

        <div className="text-center mb-4">
          <Text>OR</Text>
        </div>

        <Form.Item 
          label="User ID" 
          name="user_id"
          className="mb-4"
        >
          <Input
            name="user_id"
            className="px-3 py-2 border rounded-md w-full"
            placeholder="user_123"
            onChange={(e) => {
                e.target.value = e.target.value.trim();
              }} 
          />
        </Form.Item>

        <Form.Item 
          label="Member Role" 
          name="role"
          className="mb-4"
          rules={[
            { required: true, message: 'Please select a role!' }
          ]}
        >
          <AntSelect defaultValue="user">
            <AntSelect.Option value="admin">admin</AntSelect.Option>
            <AntSelect.Option value="user">user</AntSelect.Option>
          </AntSelect>
        </Form.Item>

        <div style={{ textAlign: "right", marginTop: "20px" }}>
          <AntButton 
            onClick={onCancel} 
            style={{ marginRight: 8 }}
          >
            Cancel
          </AntButton>
          <AntButton 
            type="primary" 
            htmlType="submit"
          >
            {mode === 'add' ? 'Add Member' : 'Save Changes'}
          </AntButton>
        </div>
      </Form>
    </Modal>
  );
};

export default TeamMemberModal;