"use client";

import React, { useState } from "react";
import { Form, Input, Button, Alert, Typography, Card, Spin, Space } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import { useLogin } from "@/app/(dashboard)/hooks/login/useLogin";

function LoginPageContent() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const { isLoading: isConfigLoading } = useUIConfig();
  const loginMutation = useLogin();

  const handleSubmit = async () => {
    loginMutation.mutate({ username, password });
  };

  const error = loginMutation.error instanceof Error ? loginMutation.error.message : null;
  const isLoading = loginMutation.isPending;

  const { Title, Text, Paragraph } = Typography;

  if (isConfigLoading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <Spin size="large" tip="Loading..." />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <Card className="w-full max-w-lg shadow-md">
        <Space direction="vertical" size="middle" className="w-full">
          <div className="text-center">
            <Title level={2}>🚅 LiteLLM</Title>
          </div>

          <div className="text-center">
            <Title level={3}>Login</Title>
            <Text type="secondary">Access your LiteLLM Admin UI.</Text>
          </div>

          <Alert
            message="Default Credentials"
            description={
              <>
                <Paragraph className="text-sm">
                  By default, Username is <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">admin</code> and
                  Password is your set LiteLLM Proxy
                  <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">MASTER_KEY</code>.
                </Paragraph>
                <Paragraph className="text-sm">
                  Need to set UI credentials or SSO?{" "}
                  <a href="https://docs.litellm.ai/docs/proxy/ui" target="_blank" rel="noopener noreferrer">
                    Check the documentation
                  </a>
                  .
                </Paragraph>
              </>
            }
            type="info"
            icon={<InfoCircleOutlined />}
            showIcon
          />

          {error && <Alert message={error} type="error" showIcon />}

          <Form onFinish={handleSubmit} layout="vertical" requiredMark={true}>
            <Form.Item
              label="Username"
              name="username"
              rules={[{ required: true, message: "Please enter your username" }]}
            >
              <Input
                placeholder="Enter your username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={isLoading}
                size="large"
                className="rounded-md border-gray-300"
              />
            </Form.Item>

            <Form.Item
              label="Password"
              name="password"
              rules={[{ required: true, message: "Please enter your password" }]}
            >
              <Input.Password
                placeholder="Enter your password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading}
                size="large"
              />
            </Form.Item>

            <Form.Item>
              <Button type="primary" htmlType="submit" loading={isLoading} disabled={isLoading} block size="large">
                {isLoading ? "Logging in..." : "Login"}
              </Button>
            </Form.Item>
          </Form>
        </Space>
      </Card>
    </div>
  );
}

export default function LoginPage() {
  const queryClient = new QueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      <LoginPageContent />
    </QueryClientProvider>
  );
}
