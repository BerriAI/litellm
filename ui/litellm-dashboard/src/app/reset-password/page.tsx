"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { changePassword, ChangePasswordResponse } from "@/components/networking";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { Alert, Button, Card, Form, Input, Space, Typography } from "antd";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function ResetPasswordPage() {
  const router = useRouter();
  const { accessToken, isLoading: isAuthLoading, isAuthorized } = useAuthorized();
  const [form] = Form.useForm();
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<ChangePasswordResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (values: {
    current_password: string;
    new_password: string;
    confirm_password: string;
  }) => {
    setError(null);
    setSuccess(null);

    if (values.new_password !== values.confirm_password) {
      setError("New password and confirmation do not match");
      return;
    }

    if (!accessToken) {
      setError("You must be logged in to change your password");
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

  const handleLogout = () => {
    clearTokenCookies();
    router.replace("/ui/login");
  };

  const handleContinue = () => {
    router.replace("/ui");
  };

  const { Title, Text } = Typography;

  if (isAuthLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <Card className="w-full max-w-lg shadow-md">
          <Space direction="vertical" size="middle" className="w-full">
            <div className="text-center">
              <Title level={2}>LiteLLM</Title>
            </div>
            <Text type="secondary">Loading...</Text>
          </Space>
        </Card>
      </div>
    );
  }

  if (!isAuthorized) {
    return null;
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <Card className="w-full max-w-lg shadow-md">
        <Space direction="vertical" size="middle" className="w-full">
          <div className="text-center">
            <Title level={2}>LiteLLM</Title>
            <Title level={3}>Change Password</Title>
          </div>

          <Alert
            message="Password Reset Required"
            description="Your administrator has required a password change. Please update your password before continuing."
            type="warning"
            showIcon
          />

          {error && <Alert message={error} type="error" showIcon />}
          {success && (
            <Alert
              message="Password Updated"
              description={success.message}
              type="success"
              showIcon
              action={
                <Button size="small" type="primary" onClick={handleContinue}>
                  Continue
                </Button>
              }
            />
          )}

          <Form form={form} onFinish={handleSubmit} layout="vertical" requiredMark={false}>
            <Form.Item
              label="Current Password"
              name="current_password"
              rules={[{ required: true, message: "Please enter your current password" }]}
            >
              <Input.Password autoComplete="current-password" size="large" />
            </Form.Item>

            <Form.Item
              label="New Password"
              name="new_password"
              rules={[
                { required: true, message: "Please enter a new password" },
                { min: 8, message: "Password must be at least 8 characters" },
              ]}
            >
              <Input.Password autoComplete="new-password" size="large" />
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
              <Input.Password autoComplete="new-password" size="large" />
            </Form.Item>

            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={isSubmitting}
                disabled={isSubmitting}
                block
                size="large"
              >
                Update Password
              </Button>
            </Form.Item>
          </Form>

          <div className="text-center">
            <Button type="link" onClick={handleLogout}>
              Log out
            </Button>
          </div>
        </Space>
      </Card>
    </div>
  );
}
