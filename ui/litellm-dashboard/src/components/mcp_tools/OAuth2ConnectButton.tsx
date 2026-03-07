"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button, message } from "antd";
import { getMcpOAuth2ConnectUrl, getMcpOAuth2Status } from "../networking";
import { MCPServer } from "./types";

interface OAuth2ConnectButtonProps {
  server: MCPServer;
  accessToken: string;
  onConnected: () => void;
}

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_MS = 10 * 60 * 1000; // 10 minutes

export const OAuth2ConnectButton: React.FC<OAuth2ConnectButtonProps> = ({
  server,
  accessToken,
  onConnected,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const popupRef = useRef<Window | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollStartRef = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    pollStartRef.current = null;
  };

  const handleConnected = () => {
    stopPolling();
    if (popupRef.current && !popupRef.current.closed) {
      popupRef.current.close();
    }
    popupRef.current = null;
    setLoading(false);
    message.success(`Connected to ${server.alias || server.server_name || "server"}`);
    onConnected();
  };

  const startPolling = () => {
    // Always clear any existing interval before starting a new one to avoid
    // double-polling if the button is clicked while a previous flow is still active.
    stopPolling();
    pollStartRef.current = Date.now();
    pollTimerRef.current = setInterval(async () => {
      // Enforce a maximum polling duration to avoid indefinite requests when
      // the popup is left open but the OAuth flow never completes.
      if (pollStartRef.current !== null && Date.now() - pollStartRef.current > MAX_POLL_MS) {
        stopPolling();
        setLoading(false);
        setError("OAuth2 connection timed out. Please try again.");
        return;
      }

      // Stop if popup was closed by user; do one final status check first so
      // that fast OAuth flows (popup auto-closes before the first poll fires)
      // are not silently missed.
      if (popupRef.current && popupRef.current.closed) {
        stopPolling();
        try {
          const finalStatus = await getMcpOAuth2Status(server.server_id, accessToken);
          if (finalStatus.connected) {
            handleConnected();
            return;
          }
        } catch {
          // ignore — popup was closed by user without completing auth
        }
        setLoading(false);
        return;
      }

      try {
        const status = await getMcpOAuth2Status(server.server_id, accessToken);
        if (status.connected) {
          handleConnected();
        }
      } catch {
        // Ignore polling errors; keep trying until popup is closed or timeout
      }
    }, POLL_INTERVAL_MS);
  };

  const handleClick = async () => {
    setError(null);
    setLoading(true);
    try {
      const { authorization_url } = await getMcpOAuth2ConnectUrl(server.server_id, accessToken);
      const popup = window.open(authorization_url, "oauth2_connect", "width=600,height=700,scrollbars=yes");
      if (!popup) {
        setError("Popup was blocked. Please allow popups for this page and try again.");
        setLoading(false);
        return;
      }
      popupRef.current = popup;
      startPolling();
    } catch (e: any) {
      setError(e.message || "Failed to start OAuth2 connection");
      setLoading(false);
    }
  };

  // Clean up on unmount: stop polling and close any open popup so it doesn't
  // stay open after the user navigates away from the page.
  useEffect(() => {
    return () => {
      stopPolling();
      if (popupRef.current && !popupRef.current.closed) {
        popupRef.current.close();
      }
      popupRef.current = null;
    };
  }, []);

  const isConnected = !!server.has_user_credential;

  return (
    <div className="flex flex-col items-start gap-1">
      <div className="flex items-center gap-2">
        {isConnected && (
          <span className="text-green-600 text-xs font-medium flex items-center gap-1">
            &#10003; Connected
          </span>
        )}
        <Button
          type={isConnected ? "default" : "primary"}
          size="small"
          loading={loading}
          onClick={handleClick}
        >
          {isConnected ? "Reconnect" : "Connect"}
        </Button>
      </div>
      {error && <span className="text-red-500 text-xs">{error}</span>}
    </div>
  );
};

export default OAuth2ConnectButton;
