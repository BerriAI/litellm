"use client";

/**
 * OAuthConnectModal
 *
 * Step-by-step OAuth2 connect modal for users connecting an OpenAPI MCP server
 * from the Chat Apps panel.  Mirrors the UX of ByokCredentialModal but initiates
 * a browser-based OAuth2 PKCE redirect instead of asking for an API key.
 */

import React, { useEffect, useState } from "react";
import { Modal } from "antd";
import {
  ArrowRightOutlined,
  ArrowLeftOutlined,
  CloseOutlined,
  CheckOutlined,
  LinkOutlined,
  LoadingOutlined,
} from "@ant-design/icons";
import { MCPServer } from "../mcp_tools/types";
import { useUserMcpOAuthFlow } from "@/hooks/useUserMcpOAuthFlow";

interface OAuthConnectModalProps {
  server: MCPServer;
  open: boolean;
  accessToken: string;
  onClose: () => void;
  onSuccess: (serverId: string) => void;
}

// Built-in permission descriptions for known providers
const GITHUB_PERMISSIONS = [
  "Read your repositories and code",
  "Manage issues and pull requests",
  "Access your profile information",
];

function getDefaultPermissions(server: MCPServer): string[] {
  const name = (server.server_name || server.alias || "").toLowerCase();
  if (name.includes("github")) return GITHUB_PERMISSIONS;
  if (server.mcp_info?.description) return [server.mcp_info.description];
  return ["Access this service on your behalf"];
}

export const OAuthConnectModal: React.FC<OAuthConnectModalProps> = ({
  server,
  open,
  accessToken,
  onClose,
  onSuccess,
}) => {
  const [step, setStep] = useState<1 | 2>(1);

  const displayName = server.alias || server.server_name || "Service";
  const firstLetter = displayName.charAt(0).toUpperCase();
  const permissions = getDefaultPermissions(server);

  // Extract scopes from credentials if present (stored as array on the server record)
  const scopes: string[] | undefined = undefined; // server-level scopes not exposed to UI

  const { startOAuthFlow, status, error } = useUserMcpOAuthFlow({
    accessToken,
    serverId: server.server_id,
    serverAlias: displayName,
    scopes,
    onSuccess: () => {
      onSuccess(server.server_id);
      handleClose();
    },
  });

  // If we return from OAuth callback, close the modal so the user isn't stuck.
  useEffect(() => {
    if (status === "success") {
      handleClose();
    }
  }, [status]);

  const handleClose = () => {
    setStep(1);
    onClose();
  };

  const isLoading = status === "authorizing" || status === "exchanging";

  const logoUrl = server.mcp_info?.logo_url;

  return (
    <Modal
      open={open}
      onCancel={handleClose}
      footer={null}
      width={480}
      closeIcon={null}
    >
      <div className="relative p-2">
        {/* Step dots + close */}
        <div className="flex items-center justify-between mb-6">
          {step === 2 ? (
            <button
              onClick={() => setStep(1)}
              className="flex items-center gap-1 text-gray-500 hover:text-gray-800 text-sm"
              disabled={isLoading}
            >
              <ArrowLeftOutlined /> Back
            </button>
          ) : (
            <div />
          )}
          <div className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${step === 1 ? "bg-blue-500" : "bg-gray-300"}`} />
            <div className={`w-2 h-2 rounded-full ${step === 2 ? "bg-blue-500" : "bg-gray-300"}`} />
          </div>
          <button onClick={handleClose} className="text-gray-400 hover:text-gray-600" disabled={isLoading}>
            <CloseOutlined />
          </button>
        </div>

        {step === 1 ? (
          <div className="text-center">
            {/* Logos */}
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow">
                L
              </div>
              <ArrowRightOutlined className="text-gray-400 text-lg" />
              <div className="w-14 h-14 rounded-xl bg-gray-100 border border-gray-200 flex items-center justify-center shadow overflow-hidden">
                {logoUrl ? (
                  <img src={logoUrl} alt={displayName} className="w-10 h-10 object-contain" />
                ) : (
                  <span className="text-gray-700 font-bold text-xl">{firstLetter}</span>
                )}
              </div>
            </div>

            <h2 className="text-2xl font-bold text-gray-900 mb-2">Connect {displayName}</h2>
            <p className="text-gray-500 mb-6">
              LiteLLM will connect to {displayName} on your behalf so you can use its tools in chat.
            </p>

            {/* How it works */}
            <div className="bg-gray-50 rounded-xl p-4 text-left mb-4">
              <div className="flex items-start gap-3">
                <div className="mt-0.5">
                  <LinkOutlined className="text-gray-400 text-base" />
                </div>
                <div>
                  <p className="font-semibold text-gray-800 mb-1">Secure OAuth 2.0</p>
                  <p className="text-gray-500 text-sm">
                    You'll be redirected to {displayName} to authorize access. LiteLLM never sees your password.
                  </p>
                </div>
              </div>
            </div>

            {/* Requested permissions */}
            <div className="bg-gray-50 rounded-xl p-4 text-left mb-6">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3">
                Requested Access
              </p>
              <ul className="space-y-2">
                {permissions.map((perm, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-gray-700">
                    <CheckOutlined className="text-green-500 flex-shrink-0" />
                    {perm}
                  </li>
                ))}
              </ul>
            </div>

            <button
              onClick={() => setStep(2)}
              className="w-full bg-gray-900 hover:bg-gray-700 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors"
            >
              Continue <ArrowRightOutlined />
            </button>
            <button
              onClick={handleClose}
              className="mt-3 w-full text-gray-400 hover:text-gray-600 text-sm py-2"
            >
              Cancel
            </button>
          </div>
        ) : (
          <div className="text-center">
            {/* Provider logo */}
            <div className="w-16 h-16 rounded-2xl bg-gray-100 border border-gray-200 flex items-center justify-center mx-auto mb-4 overflow-hidden">
              {logoUrl ? (
                <img src={logoUrl} alt={displayName} className="w-12 h-12 object-contain" />
              ) : (
                <span className="text-gray-700 font-bold text-2xl">{firstLetter}</span>
              )}
            </div>

            <h2 className="text-2xl font-bold text-gray-900 mb-2">Authorize {displayName}</h2>
            <p className="text-gray-500 mb-8">
              Click below to be redirected to {displayName} where you can authorize access.
              You'll be brought back here automatically.
            </p>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-3 mb-4 text-sm text-red-700 text-left">
                {error}
              </div>
            )}

            <button
              onClick={startOAuthFlow}
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors"
            >
              {isLoading ? (
                <>
                  <LoadingOutlined />
                  {status === "authorizing" ? "Redirecting…" : "Completing…"}
                </>
              ) : (
                <>
                  Connect with {displayName} <ArrowRightOutlined />
                </>
              )}
            </button>
            <button
              onClick={handleClose}
              disabled={isLoading}
              className="mt-3 w-full text-gray-400 hover:text-gray-600 disabled:opacity-40 text-sm py-2"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </Modal>
  );
};

export default OAuthConnectModal;
