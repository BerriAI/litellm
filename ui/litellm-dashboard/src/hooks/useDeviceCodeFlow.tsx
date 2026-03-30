import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Spin, Typography } from "antd";
import {
  type ProviderCreateInfo,
  githubCopilotInitiateAuth,
  githubCopilotCheckStatus,
  chatgptInitiateAuth,
  chatgptCheckStatus,
} from "@/components/networking";

const { Text } = Typography;

export type DeviceCodeState =
  | { phase: "idle" }
  | { phase: "polling"; deviceCode: string; userCode: string; verificationUri: string }
  | { phase: "success" }
  | { phase: "error"; message: string };

interface UseDeviceCodeFlowOptions {
  accessToken: string | null;
  providerInfo: ProviderCreateInfo | null;
  /** Called with the resolved api key and litellm provider name on successful auth. */
  onSuccess: (apiKey: string, litellmProvider: string) => void | Promise<void>;
}

export function useDeviceCodeFlow({ accessToken, providerInfo, onSuccess }: UseDeviceCodeFlowOptions) {
  const [state, setState] = useState<DeviceCodeState>({ phase: "idle" });
  const tokenRef = useRef<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const reset = useCallback(() => {
    stopPolling();
    setState({ phase: "idle" });
    tokenRef.current = null;
  }, [stopPolling]);

  const start = useCallback(async () => {
    if (!accessToken || !providerInfo) return;
    const litellmProvider = providerInfo.litellm_provider;
    const label = providerInfo.provider_display_name || litellmProvider;

    try {
      let deviceId: string;
      let userCode: string;
      let verificationUri: string;
      let pollIntervalMs: number;

      if (litellmProvider === "chatgpt") {
        const r = await chatgptInitiateAuth(accessToken);
        deviceId = r.device_auth_id;
        userCode = r.user_code;
        verificationUri = r.verification_uri;
        pollIntervalMs = r.poll_interval_ms;
      } else {
        const r = await githubCopilotInitiateAuth(accessToken);
        deviceId = r.device_code;
        userCode = r.user_code;
        verificationUri = r.verification_uri;
        pollIntervalMs = r.poll_interval_ms;
      }

      setState({ phase: "polling", deviceCode: deviceId, userCode, verificationUri });
      let interval = pollIntervalMs || 5000;

      const schedulePoll = (delayMs: number) => {
        timerRef.current = setTimeout(async () => {
          try {
            let apiKey: string | undefined;
            let failed = false;
            let errorMsg: string | undefined;
            let retryAfterMs: number | undefined;

            if (litellmProvider === "chatgpt") {
              const s = await chatgptCheckStatus(accessToken, deviceId, userCode);
              if (s.status === "complete" && s.refresh_token) {
                apiKey = s.refresh_token;
              } else if (s.status === "failed") {
                failed = true;
                errorMsg = s.error;
              }
            } else {
              const s = await githubCopilotCheckStatus(accessToken, deviceId);
              if (s.status === "complete" && s.access_token) {
                apiKey = s.access_token;
              } else if (s.status === "failed") {
                failed = true;
                errorMsg = s.error;
              } else {
                retryAfterMs = s.retry_after_ms ?? undefined;
              }
            }

            if (apiKey) {
              stopPolling();
              tokenRef.current = apiKey;
              try {
                await onSuccess(apiKey, litellmProvider);
                setState({ phase: "success" });
              } catch {
                setState({ phase: "error", message: "Authorization succeeded but callback failed" });
              }
            } else if (failed) {
              stopPolling();
              setState({ phase: "error", message: errorMsg || "Authorization failed" });
            } else {
              if (retryAfterMs != null) interval = retryAfterMs;
              schedulePoll(interval);
            }
          } catch {
            stopPolling();
            setState({ phase: "error", message: "Failed to check authorization status" });
          }
        }, delayMs);
      };
      schedulePoll(interval);
    } catch {
      setState({ phase: "error", message: `Failed to start ${label} authorization` });
    }
  }, [accessToken, providerInfo, onSuccess, stopPolling]);

  const providerLabel = providerInfo?.provider_display_name || "Provider";

  const renderUI = useCallback(() => {
    switch (state.phase) {
      case "idle":
        return (
          <div className="text-center py-2">
            <Button type="primary" onClick={start}>
              Authorize with {providerLabel}
            </Button>
          </div>
        );
      case "polling":
        return (
          <div className="text-center py-2">
            <Text className="block mb-2">Enter this code to authorize:</Text>
            <div
              style={{
                fontSize: "1.8rem",
                fontWeight: "bold",
                fontFamily: "monospace",
                letterSpacing: "0.3em",
                margin: "12px 0",
                padding: "10px 20px",
                background: "#f5f5f5",
                borderRadius: 8,
                display: "inline-block",
                userSelect: "all",
              }}
            >
              {state.userCode}
            </div>
            <div className="mb-3">
              <Button type="link" onClick={() => window.open(state.verificationUri, "_blank")}>
                Open {state.verificationUri}
              </Button>
            </div>
            <Spin />
            <Text className="block mt-2 mb-3" type="secondary">
              Waiting for {providerLabel} authorization...
            </Text>
            <Button onClick={reset}>Cancel</Button>
          </div>
        );
      case "success":
        return (
          <div className="text-center py-2">
            <Text type="success" className="block mb-2">
              Authorization complete!
            </Text>
          </div>
        );
      case "error":
        return (
          <div className="text-center py-2">
            <Text type="danger" className="block mb-3">
              {state.message}
            </Text>
            <Button style={{ marginRight: 8 }} onClick={() => setState({ phase: "idle" })}>
              Retry
            </Button>
            <Button onClick={reset}>Cancel</Button>
          </div>
        );
    }
  }, [state, start, reset, providerLabel]);

  return { state, start, reset, renderUI };
}
