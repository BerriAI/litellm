import { changePassword, ChangePasswordResponse } from "@/components/networking";
import { Alert, Button, Form, Input, Modal } from "antd";
import React, { useState } from "react";

interface ChangePasswordModalProps {
  accessToken: string;
  open: boolean;
  onClose: () => void;
}

const ChangePasswordModal: React.FC<ChangePasswordModalProps> = ({ accessToken, open, onClose }) => {
  const [form] = Form.useForm();
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<ChangePasswordResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (values: { current_password: string; new_password: string; confirm_password: string }) => {
    setError(null);
    setSuccess(null);

    if (values.new_password !== values.confirm_password) {
      setError("New password and confirmation do not match");
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await changePassword(accessToken, {
        current_password: values.current_password,
        new_password: values.new_password,
      });
      setSuccess(response);
      form.resetFields();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to change password");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    form.resetFields();
    setError(null);
    setSuccess(null);
    onClose();
  };

  return (
    <Modal title="Change Password" open={open} onCancel={handleClose} footer={null} destroyOnClose>
      {error && <Alert message={error} type="error" showIcon className="mb-4" />}
      {success && <Alert message={success.message} type="success" showIcon className="mb-4" />}
      <Form form={form} onFinish={handleSubmit} layout="vertical" requiredMark={false}>
        <Form.Item
          label="Current Password"
          name="current_password"
          rules={[{ required: true, message: "Please enter your current password" }]}
        >
          <Input.Password autoComplete="current-password" />
        </Form.Item>

        <Form.Item
          label="New Password"
          name="new_password"
          rules={[
            { required: true, message: "Please enter a new password" },
            { min: 8, message: "Password must be at least 8 characters" },
          ]}
        >
          <Input.Password autoComplete="new-password" />
        </Form.Item>

        <Form.Item
          label="Confirm New Password"
          name="confirm_password"
          rules={[
            { required: true, message: "Please confirm your new password" },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue("new_password") === value) {
                  return Promise.resolve();
                }
                return Promise.reject(new Error("Passwords do not match"));
              },
            }),
          ]}
        >
          <Input.Password autoComplete="new-password" />
        </Form.Item>

        <div className="flex justify-end gap-2">
          <Button onClick={handleClose} disabled={isSubmitting}>
            {success ? "Close" : "Cancel"}
          </Button>
          {!success && (
            <Button type="primary" htmlType="submit" loading={isSubmitting} disabled={isSubmitting}>
              Update Password
            </Button>
          )}
        </div>
      </Form>
    </Modal>
  );
};

export default ChangePasswordModal;
