"use client";

import React, { useState } from "react";
import { Modal, Input, Switch, message } from "antd";
import {
  KeyOutlined,
  LockOutlined,
  CheckOutlined,
  ArrowRightOutlined,
  ArrowLeftOutlined,
  CloseOutlined,
  LinkOutlined,
} from "@ant-design/icons";
import { MCPServer } from "./types";

interface ByokCredentialModalProps {
  server: MCPServer;
  open: boolean;
  onClose: () => void;
  onSuccess: (serverId: string) => void;
  accessToken: string;
}

export const ByokCredentialModal: React.FC<ByokCredentialModalProps> = ({
  server,
  open,
  onClose,
  onSuccess,
  accessToken,
}) => {
  const [step, setStep] = useState<1 | 2>(1);
  const [apiKey, setApiKey] = useState("");
  const [saveKey, setSaveKey] = useState(true);
  const [loading, setLoading] = useState(false);

  const serverDisplayName = server.alias || server.server_name || "Service";
  const firstLetter = serverDisplayName.charAt(0).toUpperCase();

  const handleClose = () => {
    setStep(1);
    setApiKey("");
    setSaveKey(true);
    setLoading(false);
    onClose();
  };

  const handleAuthorize = async () => {
    if (!apiKey.trim()) {
      message.error("Please enter your API key");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`/v1/mcp/server/${server.server_id}/user-credential`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ credential: apiKey.trim(), save: saveKey }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err?.detail?.error || "Failed to save credential");
      }
      message.success(`Connected to ${serverDisplayName}`);
      onSuccess(server.server_id);
      handleClose();
    } catch (e: any) {
      message.error(e.message || "Failed to connect");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      open={open}
      onCancel={handleClose}
      footer={null}
      width={480}
      closeIcon={null}
      className="byok-modal"
    >
      <div className="relative p-2">
        {/* Step dots + close */}
        <div className="flex items-center justify-between mb-6">
          {step === 2 ? (
            <button
              onClick={() => setStep(1)}
              className="flex items-center gap-1 text-gray-500 hover:text-gray-800 text-sm"
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
          <button onClick={handleClose} className="text-gray-400 hover:text-gray-600">
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
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow">
                {firstLetter}
              </div>
            </div>

            <h2 className="text-2xl font-bold text-gray-900 mb-2">Connect {serverDisplayName}</h2>
            <p className="text-gray-500 mb-6">
              LiteLLM needs access to {serverDisplayName} to complete your request.
            </p>

            {/* How it works */}
            <div className="bg-gray-50 rounded-xl p-4 text-left mb-4">
              <div className="flex items-start gap-3">
                <div className="mt-0.5">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gray-500">
                    <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2" />
                    <path d="M8 4v16M16 4v16" stroke="currentColor" strokeWidth="2" />
                  </svg>
                </div>
                <div>
                  <p className="font-semibold text-gray-800 mb-1">How it works</p>
                  <p className="text-gray-500 text-sm">
                    LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to{" "}
                    {serverDisplayName}&apos;s API.
                  </p>
                </div>
              </div>
            </div>

            {/* Requested access */}
            {server.byok_description && server.byok_description.length > 0 && (
              <div className="bg-gray-50 rounded-xl p-4 text-left mb-6">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-green-500">
                    <path
                      d="M12 2L12 22M2 12L22 12"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
                  </svg>
                  Requested Access
                </p>
                <ul className="space-y-2">
                  {server.byok_description.map((item, i) => (
                    <li key={i} className="flex items-center gap-2 text-sm text-gray-700">
                      <CheckOutlined className="text-green-500 flex-shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <button
              onClick={() => setStep(2)}
              className="w-full bg-gray-900 hover:bg-gray-700 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors"
            >
              Continue to Authentication <ArrowRightOutlined />
            </button>
            <button
              onClick={handleClose}
              className="mt-3 w-full text-gray-400 hover:text-gray-600 text-sm py-2"
            >
              Cancel
            </button>
          </div>
        ) : (
          <div>
            {/* Key icon */}
            <div className="w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center mb-4">
              <KeyOutlined className="text-blue-400 text-xl" />
            </div>

            <h2 className="text-2xl font-bold text-gray-900 mb-2">Provide API Key</h2>
            <p className="text-gray-500 mb-6">
              Enter your {serverDisplayName} API key to authorize this connection.
            </p>

            <div className="mb-4">
              <label className="block text-sm font-semibold text-gray-800 mb-2">
                {serverDisplayName} API Key
              </label>
              <Input.Password
                placeholder="Enter your API key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                size="large"
                className="rounded-lg"
              />
              {server.byok_api_key_help_url && (
                <a
                  href={server.byok_api_key_help_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:text-blue-700 text-sm mt-2 flex items-center gap-1"
                >
                  Where do I find my API key? <LinkOutlined />
                </a>
              )}
            </div>

            {/* Save toggle */}
            <div className="bg-gray-50 rounded-xl p-4 flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gray-500">
                  <path
                    d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"
                    fill="currentColor"
                  />
                </svg>
                <span className="text-sm font-medium text-gray-800">Save key for future use</span>
              </div>
              <Switch checked={saveKey} onChange={setSaveKey} />
            </div>

            {/* Security note */}
            <div className="bg-blue-50 rounded-xl p-4 flex items-start gap-3 mb-6">
              <LockOutlined className="text-blue-400 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-blue-700">
                Your key is stored securely and transmitted over HTTPS. It is never shared with third parties.
              </p>
            </div>

            <button
              onClick={handleAuthorize}
              disabled={loading}
              className="w-full bg-blue-500 hover:bg-blue-600 disabled:opacity-60 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors"
            >
              <LockOutlined /> Connect &amp; Authorize
            </button>
          </div>
        )}
      </div>
    </Modal>
  );
};

export default ByokCredentialModal;
