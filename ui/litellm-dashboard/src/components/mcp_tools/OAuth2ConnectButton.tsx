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

export const OAuth2ConnectButton: React.FC<OAuth2ConnectButtonProps> = ({
  server,
  accessToken,
  onConnected,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const popupRef = useRef<Window | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => {
    if (pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
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
    pollTimerRef.current = setInterval(async () => {
      // Stop if popup was closed by user
      if (popupRef.current && popupRef.current.closed) {
        stopPolling();
        setLoading(false);
        return;
      }

      try {
        const status = await getMcpOAuth2Status(server.server_id, accessToken);
        if (status.connected) {
          handleConnected();
        }
      } catch {
        // Ignore polling errors; keep trying until popup is closed
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

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopPolling();
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
