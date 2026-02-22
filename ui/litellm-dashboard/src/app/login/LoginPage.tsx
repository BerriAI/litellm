"use client";

import { useLogin } from "@/app/(dashboard)/hooks/login/useLogin";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { getProxyBaseUrl } from "@/components/networking";
import { getCookie } from "@/utils/cookieUtils";
import { isJwtExpired } from "@/utils/jwtUtils";
import { InfoCircleOutlined } from "@ant-design/icons";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Alert, Button, Card, Form, Input, Popover, Space, Typography } from "antd";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

function LoginPageContent() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const { data: uiConfig, isLoading: isConfigLoading } = useUIConfig();
  const loginMutation = useLogin();
  const router = useRouter();

  useEffect(() => {
    if (isConfigLoading) {
      return;
    }

    // Check if admin UI is disabled
    if (uiConfig && uiConfig.admin_ui_disabled) {
      setIsLoading(false);
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

  // Show disabled message if admin UI is disabled
  if (uiConfig && uiConfig.admin_ui_disabled) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <Card className="w-full max-w-lg shadow-md">
          <Space direction="vertical" size="middle" className="w-full">
            <div className="text-center">
              <Title level={2}>🚅 LiteLLM</Title>
            </div>

            <Alert
              message="Admin UI Disabled"
              description={
                <>
                  <Paragraph className="text-sm">
                    The Admin UI has been disabled by the administrator. To re-enable it, please update the following
                    environment variable:
                  </Paragraph>
                  <Paragraph className="text-sm">
                    <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">DISABLE_ADMIN_UI=False</code>
                  </Paragraph>
                </>
              }
              type="warning"
              showIcon
            />
          </Space>
        </Card>
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
                disabled={isLoginLoading}
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
            <Form.Item>
              {!uiConfig?.sso_configured ? (
                <Popover
                  content="Please configure SSO to log in with SSO."
                  trigger="hover"
                >
                  <Button disabled block size="large">
                    Login with SSO
                  </Button>
                </Popover>
              ) : (
                <Button
                  disabled={isLoginLoading}
                  onClick={() =>
                    router.push(`${getProxyBaseUrl()}/sso/key/generate`)
                  }
                  block
                  size="large"
                >
                  Login with SSO
                </Button>
              )}
            </Form.Item>
          </Form>
        </Space>
        {uiConfig?.sso_configured && (
          <Alert
            type="info"
            showIcon
            closable
            message={<Text>Single Sign-On (SSO) is enabled. LiteLLM no longer automatically redirects to the SSO login flow upon loading this page. To re-enable auto-redirect-to-SSO, set <Text code>AUTO_REDIRECT_UI_LOGIN_TO_SSO=true</Text> in your environment configuration.</Text>}
          />
        )}
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
