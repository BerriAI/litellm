"use client";

import React from "react";
import { Alert, Button, Card, Form, Input, Typography } from "antd";
import { useForgotPassword } from "@/app/(dashboard)/hooks/passwordReset/usePasswordReset";
import { getLoginUrl } from "@/utils/returnUrlUtils";

export function ForgotPasswordForm() {
  const { mutate: submitForgotPassword, isPending, isSuccess, error } = useForgotPassword();

  const handleSubmit = (values: { email: string }) => {
    submitForgotPassword(values.email);
  };

  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Card>
        <Typography.Title level={5} className="text-center mb-5">
          🚅 LiteLLM
        </Typography.Title>
        <Typography.Title level={3}>Forgot Password</Typography.Title>
        <Typography.Text>Enter your email address and we will send you a link to reset your password.</Typography.Text>

        {isSuccess ? (
          <Alert
            className="mt-4"
            type="success"
            message="If an account exists for this email, a password reset link has been sent."
            showIcon
          />
        ) : (
          <Form className="mt-10 mb-5" layout="vertical" onFinish={handleSubmit}>
            <Form.Item
              label="Email Address"
              name="email"
              rules={[{ required: true, type: "email", message: "Please enter a valid email address" }]}
            >
              <Input type="email" />
            </Form.Item>

            {error && <Alert type="error" message={(error as Error).message} showIcon className="mb-4" />}

            <div className="mt-10">
              <Button htmlType="submit" loading={isPending}>
                Send Reset Link
              </Button>
            </div>
          </Form>
        )}

        <div className="mt-4">
          <Button type="link" href={getLoginUrl()}>
            Back to Login
          </Button>
        </div>
      </Card>
    </div>
  );
}
