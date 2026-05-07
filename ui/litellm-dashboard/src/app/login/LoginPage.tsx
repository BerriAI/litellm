"use client";

import { useLogin } from "@/app/(dashboard)/hooks/login/useLogin";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { exchangeLoginCode, getProxyBaseUrl, switchToWorkerUrl } from "@/components/networking";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { isJwtExpired } from "@/utils/jwtUtils";
import { consumeReturnUrl, getReturnUrl, isValidReturnUrl } from "@/utils/returnUrlUtils";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { useRouter } from "next/navigation";
import type { FormEvent } from "react";
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

  const handleSubmit = (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();

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

  if (isConfigLoading || isLoading) {
    return <LoadingScreen />;
  }

  // Show disabled message if admin UI is disabled
  if (uiConfig && uiConfig.admin_ui_disabled) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4">
        <Card className="w-full max-w-lg shadow-md">
          <CardHeader className="items-center text-center">
            <CardTitle className="text-3xl">🚅 LiteLLM</CardTitle>
            <CardDescription>Admin dashboard</CardDescription>
          </CardHeader>
          <CardContent>
            <Alert variant="warning">
              <AlertTitle>Admin UI Disabled</AlertTitle>
              <AlertDescription>
                <p>
                  The Admin UI has been disabled by the administrator. To re-enable it, please update the following
                  environment variable:
                </p>
                <p className="mt-2">
                  <code className="rounded bg-white/70 px-1.5 py-0.5 text-xs">DISABLE_ADMIN_UI=False</code>
                </p>
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4 py-10">
      <Card className="w-full max-w-lg shadow-md">
        <CardHeader className="items-center text-center">
          <CardTitle className="text-3xl">🚅 LiteLLM</CardTitle>
          <CardDescription>Access your LiteLLM Admin UI.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="text-center">
            <h1 className="text-2xl font-semibold tracking-tight text-slate-950">Login</h1>
          </div>

          <Alert variant="info">
            <AlertTitle>Default Credentials</AlertTitle>
            <AlertDescription>
              <p>
                By default, Username is <code className="rounded bg-white/70 px-1.5 py-0.5 text-xs">admin</code> and
                Password is your set LiteLLM Proxy{" "}
                <code className="rounded bg-white/70 px-1.5 py-0.5 text-xs">MASTER_KEY</code>.
              </p>
              <p className="mt-2">
                Need to set UI credentials or SSO?{" "}
                <a
                  className="font-medium underline underline-offset-4"
                  href="https://docs.litellm.ai/docs/proxy/ui"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Check the documentation
                </a>
                .
              </p>
            </AlertDescription>
          </Alert>

          {error && (
            <Alert variant="destructive">
              <AlertTitle>Login failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {uiConfig?.is_control_plane && workers.length > 0 && (
              <div className="space-y-2">
                <Label htmlFor="worker">Worker</Label>
                <Select
                  id="worker"
                  value={selectedWorkerId || ""}
                  onChange={(event) => setSelectedWorkerId(event.target.value || null)}
                  placeholder="Choose a worker to connect to"
                  options={workers.map((w) => ({
                    label: w.name,
                    value: w.worker_id,
                  }))}
                />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                name="username"
                placeholder="Enter your username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={isLoginLoading}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                placeholder="Enter your password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoginLoading}
                required
              />
            </div>

            <Button type="submit" loading={isLoginLoading} disabled={isLoginLoading} block size="lg">
              {isLoginLoading ? "Logging in..." : "Login"}
            </Button>
            {!uiConfig?.sso_configured ? (
              <Button
                type="button"
                disabled
                block
                size="lg"
                variant="outline"
                title="Please configure SSO to log in with SSO."
              >
                Login with SSO
              </Button>
            ) : (
              <Button
                type="button"
                variant="outline"
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
                size="lg"
              >
                Login with SSO
              </Button>
            )}
          </form>
        </CardContent>
        {uiConfig?.sso_configured && (
          <Alert
            variant="info"
            className="mx-6 mb-6"
          >
            <AlertDescription>
              Single Sign-On (SSO) is enabled. LiteLLM no longer automatically redirects to the SSO login flow upon
              loading this page. To re-enable auto-redirect-to-SSO, set{" "}
              <code className="rounded bg-white/70 px-1.5 py-0.5 text-xs">AUTO_REDIRECT_UI_LOGIN_TO_SSO=true</code> in
              your environment configuration.
            </AlertDescription>
          </Alert>
        )}
      </Card>
    </div>
  );
}

export default function LoginPage() {
  return <LoginPageContent />;
}
