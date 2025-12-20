"use client";

import { useLogin } from "@/app/(dashboard)/hooks/login/useLogin";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { getProxyBaseUrl } from "@/components/networking";
import { getCookie } from "@/utils/cookieUtils";
import { isJwtExpired } from "@/utils/jwtUtils";
import { InfoCircleOutlined } from "@ant-design/icons";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Alert, Button, Card, ConfigProvider, Form, Input, Space, Typography } from "antd";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ThemeProvider, useTheme } from "@/contexts/ThemeContext";
import { getAntdTheme } from "@/config/antdTheme";

function LoginPageContent() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const { data: uiConfig, isLoading: isConfigLoading } = useUIConfig();
  const loginMutation = useLogin();
  const router = useRouter();
  const { isDarkMode } = useTheme();

  useEffect(() => {
    if (isConfigLoading) {
      return;
    }

    const rawToken = getCookie("token");
    if (rawToken && !isJwtExpired(rawToken)) {
      router.replace(`${getProxyBaseUrl()}/ui`);
      return;
    }

    if (uiConfig && uiConfig.auto_redirect_to_sso) {
      router.push(`${getProxyBaseUrl()}/sso/key/generate`);
      return;
    }

    setIsLoading(false);
  }, [isConfigLoading, router, uiConfig]);

  const handleSubmit = () => {
    loginMutation.mutate(
      { username, password },
      {
        onSuccess: (data) => {
          router.push(data.redirect_url);
        },
      },
    );
  };

  const error = loginMutation.error instanceof Error ? loginMutation.error.message : null;
  const isLoginLoading = loginMutation.isPending;

  const { Title, Text, Paragraph } = Typography;

  if (isConfigLoading || isLoading) {
    return <LoadingScreen />;
  }

  return (
    <ConfigProvider theme={getAntdTheme(isDarkMode)}>
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-[#141414]">
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
                    By default, Username is <code className="bg-gray-100 dark:bg-[#252525] px-1 py-0.5 rounded text-xs">admin</code> and
                    Password is your set LiteLLM Proxy
                    <code className="bg-gray-100 dark:bg-[#252525] px-1 py-0.5 rounded text-xs">MASTER_KEY</code>.
                  </Paragraph>
                  <Paragraph className="text-sm">
                    Need to set UI credentials or SSO?{" "}
                    <a href="https://docs.litellm.ai/docs/proxy/ui" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-400">
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
                  disabled={isLoginLoading}
                  size="large"
                  className="rounded-md border-gray-300 dark:border-[#2a2a2a]"
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
                  disabled={isLoginLoading}
                  size="large"
                />
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={isLoginLoading}
                  disabled={isLoginLoading}
                  block
                  size="large"
                >
                  {isLoginLoading ? "Logging in..." : "Login"}
                </Button>
              </Form.Item>
            </Form>
          </Space>
        </Card>
      </div>
    </ConfigProvider>
  );
}

export default function LoginPage() {
  const queryClient = new QueryClient();

  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <LoginPageContent />
      </QueryClientProvider>
    </ThemeProvider>
  );
}
