"use client";

import { useLogin } from "@/app/(dashboard)/hooks/login/useLogin";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { exchangeLoginCode, getProxyBaseUrl, switchToWorkerUrl } from "@/components/networking";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { isJwtExpired } from "@/utils/jwtUtils";
import { consumeReturnUrl, getReturnUrl, isValidReturnUrl } from "@/utils/returnUrlUtils";
import { InfoCircleOutlined, CloudServerOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Form, Input, Popover, Select, Space, Typography } from "antd";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useWorker } from "@/hooks/useWorker";

function LoginPageContent() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const { data: uiConfig, isLoading: isConfigLoading } = useUIConfig();
  const loginMutation = useLogin();
  const router = useRouter();
  const { workers, selectWorker } = useWorker();
  const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null);

  // Pre-select worker from URL param (e.g. /ui/login?worker=team-b)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const workerParam = params.get("worker");
    if (workerParam) {
      setSelectedWorkerId(workerParam);
    }
  }, []);

  useEffect(() => {
    if (isConfigLoading) {
      return;
    }

    // Check if admin UI is disabled
    if (uiConfig && uiConfig.admin_ui_disabled) {
      setIsLoading(false);
      return;
    }

    // Cross-origin SSO: worker redirected back with a single-use code.
    // Exchange it for the JWT via the worker's /v3/login/exchange endpoint.
    const params = new URLSearchParams(window.location.search);
    const rawSsoCode = params.get("code");
    // Validate the SSO code is a plausible OAuth authorization code (alphanumeric
    // plus common URL-safe chars) so that arbitrary user input cannot trigger the
    // exchange endpoint.
    const ssoCode =
      rawSsoCode && /^[a-zA-Z0-9._~+/=-]+$/.test(rawSsoCode) ? rawSsoCode : null;
    if (ssoCode) {
      const rawWorkerUrl = localStorage.getItem("litellm_worker_url");
      // Validate the stored worker URL: only allow http(s) URLs.
      const workerUrl =
        rawWorkerUrl && /^https?:\/\/.+/.test(rawWorkerUrl) ? rawWorkerUrl : null;
      exchangeLoginCode(ssoCode, workerUrl).then(() => {
        params.delete("code");
        const cleanSearch = params.toString();
        window.history.replaceState(null, "", window.location.pathname + (cleanSearch ? `?${cleanSearch}` : ""));
        router.replace("/ui/?login=success");
      });
      return;
    }

    // Backwards compat: handle direct token in URL (legacy flow)
    const urlToken = params.get("token");
    if (urlToken && !isJwtExpired(urlToken)) {
      document.cookie = `token=${urlToken}; path=/; SameSite=Lax`;
      params.delete("token");
      const cleanSearch = params.toString();
      window.history.replaceState(
        null,
        "",
        window.location.pathname + (cleanSearch ? `?${cleanSearch}` : ""),
      );
      router.replace("/ui/?login=success");
      return;
    }

    // If switching workers on a control plane, clear the old token and show login
    const switchingWorker = params.has("worker");
    if (switchingWorker && uiConfig?.is_control_plane) {
      clearTokenCookies();
      setIsLoading(false);
      return;
    }

    const rawToken = getCookie("token");
    if (rawToken && !isJwtExpired(rawToken)) {
      // User already logged in - redirect to return URL or default
      const returnUrl = consumeReturnUrl();
      if (returnUrl) {
        router.replace(returnUrl);
      } else {
        router.replace("/ui");
      }
      return;
    }

    if (uiConfig && uiConfig.auto_redirect_to_sso) {
      // For SSO, pass the return URL to the SSO endpoint
      const returnUrl = getReturnUrl();
      let ssoUrl = `${getProxyBaseUrl()}/sso/key/generate`;
      if (returnUrl && isValidReturnUrl(returnUrl)) {
        ssoUrl += `?redirect_to=${encodeURIComponent(returnUrl)}`;
      }
      router.push(ssoUrl);
      return;
    }

    setIsLoading(false);
  }, [isConfigLoading, router, uiConfig]);

  const handleSubmit = () => {
    // If a worker is selected, point proxyBaseUrl at it before login
    const selectedWorker = workers.find((w) => w.worker_id === selectedWorkerId);
    if (selectedWorker) {
      switchToWorkerUrl(selectedWorker.url);
    }

    loginMutation.mutate(
      { username, password, useV3: !!selectedWorker },
      {
        onSuccess: (data) => {
          // Update the worker context with the selected worker
          if (selectedWorker) {
            selectWorker(selectedWorker.worker_id);
            // Stay on the CP's UI — proxyBaseUrl already points at the worker
            router.push("/ui/?login=success");
          } else {
            // Normal (non-control-plane) login — follow the server's redirect
            const returnUrl = consumeReturnUrl();
            if (returnUrl) {
              router.push(returnUrl);
            } else {
              router.push(data.redirect_url);
            }
          }
        },
        onError: () => {
          // Reset proxyBaseUrl on login failure
          if (selectedWorker) {
            switchToWorkerUrl(null);
          }
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

          <Form onFinish={handleSubmit} layout="vertical" requiredMark={false}>
            {uiConfig?.is_control_plane && workers.length > 0 && (
              <Form.Item label="Worker" style={{ marginBottom: 16 }}>
                <Select
                  value={selectedWorkerId || undefined}
                  onChange={(value) => setSelectedWorkerId(value)}
                  placeholder="Choose a worker to connect to"
                  size="large"
                  suffixIcon={<CloudServerOutlined />}
                  options={workers.map((w) => ({
                    label: w.name,
                    value: w.worker_id,
                  }))}
                />
              </Form.Item>
            )}

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
                  disabled={isLoginLoading || (!!selectedWorkerId && workers.length === 0)}
                  onClick={() => {
                    const selectedWorker = workers.find((w) => w.worker_id === selectedWorkerId);
                    if (selectedWorker) {
                      // Store worker selection so useWorker hook restores it after redirect
                      localStorage.setItem("litellm_selected_worker_id", selectedWorkerId!);
                      switchToWorkerUrl(selectedWorker.url);
                    }
                    // SSO on the worker (or this instance if no worker), always
                    // include return_to so the callback redirects back here
                    const ssoBase = selectedWorker?.url ?? getProxyBaseUrl();
                    const returnTo = encodeURIComponent(window.location.origin + "/ui/login");
                    router.push(`${ssoBase}/sso/key/generate?return_to=${returnTo}`);
                  }}
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
  return <LoginPageContent />;
}
