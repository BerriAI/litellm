import { Alert, Button, Space, Spin, Typography } from "antd";
import React, { useCallback, useEffect, useRef, useState } from "react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type {
  ChatGPTOAuthStartResponse,
  ChatGPTOAuthStatusResponse,
} from "@/components/networking";

const { Text, Link, Paragraph } = Typography;

const POLL_INTERVAL_MS = 3000;

export interface OAuthDeviceLoginButtonProps {
  providerLabel: string; // e.g. "ChatGPT" or "GitHub Copilot"
  credentialName?: string;
  onSuccess: () => void;
  startCall: (accessToken: string, credentialName: string) => Promise<ChatGPTOAuthStartResponse>;
  statusCall: (accessToken: string, sessionId: string) => Promise<ChatGPTOAuthStatusResponse>;
  cancelCall: (accessToken: string, sessionId: string) => Promise<void>;
}

type Phase =
  | { kind: "idle" }
  | {
      kind: "active";
      sessionId: string;
      userCode: string;
      verificationUrl: string;
    }
  | { kind: "success" }
  | { kind: "error"; message: string };

const OAuthDeviceLoginButton: React.FC<OAuthDeviceLoginButtonProps> = ({
  providerLabel,
  credentialName,
  onSuccess,
  startCall,
  statusCall,
  cancelCall,
}) => {
  const { accessToken } = useAuthorized();
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const startLogin = useCallback(async () => {
    if (!accessToken || !credentialName) return;
    setPhase({ kind: "idle" });
    try {
      const start = await startCall(accessToken, credentialName);
      setPhase({
        kind: "active",
        sessionId: start.session_id,
        userCode: start.user_code,
        verificationUrl: start.verification_url,
      });

      stopPolling();
      pollTimer.current = setInterval(async () => {
        try {
          const status = await statusCall(accessToken, start.session_id);
          if (status.status === "success") {
            stopPolling();
            setPhase({ kind: "success" });
            onSuccess();
          } else if (status.status === "error" || status.status === "cancelled") {
            stopPolling();
            setPhase({
              kind: "error",
              message: status.message || status.status,
            });
          }
        } catch {
          // transient error — keep polling
        }
      }, POLL_INTERVAL_MS);
    } catch (err) {
      setPhase({
        kind: "error",
        message: err instanceof Error ? err.message : "Failed to start login",
      });
    }
  }, [accessToken, credentialName, onSuccess, startCall, statusCall, stopPolling]);

  const cancel = useCallback(async () => {
    if (phase.kind !== "active" || !accessToken) return;
    stopPolling();
    try {
      await cancelCall(accessToken, phase.sessionId);
    } catch {
      // ignore — cancel is best-effort
    }
    setPhase({ kind: "idle" });
  }, [accessToken, phase, cancelCall, stopPolling]);

  if (phase.kind === "active") {
    return (
      <Space direction="vertical" style={{ width: "100%" }} size="middle">
        <Alert
          type="info"
          showIcon
          message={`Sign in with ${providerLabel}`}
          description={
            <Space direction="vertical" size="small">
              <Paragraph style={{ marginBottom: 0 }}>
                Open the link below in a new tab and enter this code:
              </Paragraph>
              <Text code copyable style={{ fontSize: "1.4em", letterSpacing: "0.1em" }}>
                {phase.userCode}
              </Text>
              <Link href={phase.verificationUrl} target="_blank" rel="noreferrer">
                {phase.verificationUrl}
              </Link>
              <Space>
                <Spin size="small" />
                <Text type="secondary">Waiting for browser confirmation…</Text>
              </Space>
            </Space>
          }
        />
        <Button onClick={cancel}>Cancel</Button>
      </Space>
    );
  }

  if (phase.kind === "success") {
    return (
      <Alert
        type="success"
        showIcon
        message="Signed in"
        description="OAuth credential stored successfully."
      />
    );
  }

  return (
    <Space direction="vertical" style={{ width: "100%" }} size="middle">
      {phase.kind === "error" && (
        <Alert type="error" showIcon message={phase.message} />
      )}
      <Paragraph type="secondary" style={{ marginBottom: 0 }}>
        Enter a credential name above, then click Sign in to receive a short
        code to enter in your browser.
      </Paragraph>
      <Button type="primary" onClick={startLogin} disabled={!credentialName}>
        Sign in with {providerLabel}
      </Button>
    </Space>
  );
};

export default OAuthDeviceLoginButton;
