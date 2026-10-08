"use client";

import React, { useId, useState } from "react";
import { toast } from "@/lib/toast";
import { fetchClient } from "@/lib/http/api";
import { ApiError } from "@/lib/http/client";
import { ArrowLeft, ArrowRight, Check, Key, Link2, Lock, X } from "lucide-react";
import { MCPServer } from "./types";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Switch } from "@/components/ui/switch";

const byokSaveErrorMessage = (e: unknown): string => {
  if (e instanceof ApiError) {
    const detail = (e.body as { detail?: { error?: string } } | null)?.detail?.error;
    if (detail) return detail;
  }
  return e instanceof Error && e.message ? e.message : "Failed to connect";
};

interface ByokCredentialModalProps {
  server: MCPServer;
  open: boolean;
  onClose: () => void;
  onSuccess: (serverId: string) => void;
}

export const ByokCredentialModal: React.FC<ByokCredentialModalProps> = ({ server, open, onClose, onSuccess }) => {
  const [step, setStep] = useState<1 | 2>(1);
  const [apiKey, setApiKey] = useState("");
  const [saveKey, setSaveKey] = useState(true);
  const [loading, setLoading] = useState(false);
  const apiKeyInputId = useId();

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
      toast.error("Please enter your API key");
      return;
    }
    setLoading(true);
    try {
      await fetchClient.POST("/v1/mcp/server/{server_id}/user-credential", {
        params: { path: { server_id: server.server_id } },
        body: { credential: apiKey.trim(), save: saveKey },
      });
      toast.success(`Connected to ${serverDisplayName}`);
      onSuccess(server.server_id);
      handleClose();
    } catch (e) {
      toast.error(byokSaveErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent
        className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[480px] byok-modal"
        showCloseButton={false}
      >
        <div className="relative p-2">
          {/* Step dots + close */}
          <div className="flex items-center justify-between mb-6">
            {step === 2 ? (
              <button
                onClick={() => setStep(1)}
                className="flex items-center gap-1 text-muted-foreground hover:text-foreground text-sm"
              >
                <ArrowLeft className="size-3.5" /> Back
              </button>
            ) : (
              <div />
            )}
            <div className="flex items-center gap-1.5">
              <div className={`w-2 h-2 rounded-full ${step === 1 ? "bg-info" : "bg-border"}`} />
              <div className={`w-2 h-2 rounded-full ${step === 2 ? "bg-info" : "bg-border"}`} />
            </div>
            <button onClick={handleClose} className="text-muted-foreground hover:text-foreground">
              <X className="size-4" />
            </button>
          </div>

          {step === 1 ? (
            <div className="text-center">
              {/* Logos */}
              <div className="flex items-center justify-center gap-3 mb-6">
                <div className="w-14 h-14 rounded-xl bg-linear-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow-sm">
                  L
                </div>
                <ArrowRight className="size-4.5 text-muted-foreground" />
                <div className="w-14 h-14 rounded-xl bg-linear-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow-sm">
                  {firstLetter}
                </div>
              </div>

              <h2 className="text-2xl font-bold text-foreground mb-2">Connect {serverDisplayName}</h2>
              <p className="text-muted-foreground mb-6">
                LiteLLM needs access to {serverDisplayName} to complete your request.
              </p>

              {/* How it works */}
              <div className="bg-muted rounded-xl p-4 text-left mb-4">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-muted-foreground">
                      <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2" />
                      <path d="M8 4v16M16 4v16" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-semibold text-foreground mb-1">How it works</p>
                    <p className="text-muted-foreground text-sm">
                      LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to{" "}
                      {serverDisplayName}&apos;s API.
                    </p>
                  </div>
                </div>
              </div>

              {/* Requested access */}
              {server.byok_description && server.byok_description.length > 0 && (
                <div className="bg-muted rounded-xl p-4 text-left mb-6">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-success">
                      <path d="M12 2L12 22M2 12L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
                    </svg>
                    Requested Access
                  </p>
                  <ul className="space-y-2">
                    {server.byok_description.map((item, i) => (
                      <li key={i} className="flex items-center gap-2 text-sm text-foreground">
                        <Check className="size-3.5 shrink-0 text-success" />
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
                Continue to Authentication <ArrowRight className="size-4" />
              </button>
              <button
                onClick={handleClose}
                className="mt-3 w-full text-muted-foreground hover:text-foreground text-sm py-2"
              >
                Cancel
              </button>
            </div>
          ) : (
            <div>
              {/* Key icon */}
              <div className="w-12 h-12 rounded-full bg-info/10 flex items-center justify-center mb-4">
                <Key className="size-5 text-info" />
              </div>

              <h2 className="text-2xl font-bold text-foreground mb-2">Provide API Key</h2>
              <p className="text-muted-foreground mb-6">
                Enter your {serverDisplayName} API key to authorize this connection.
              </p>

              <div className="mb-4">
                <label htmlFor={apiKeyInputId} className="block text-sm font-semibold text-foreground mb-2">
                  {serverDisplayName} API Key
                </label>
                <PasswordInput
                  id={apiKeyInputId}
                  placeholder="Enter your API key"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  groupClassName="rounded-lg"
                />
                {server.byok_api_key_help_url && (
                  <a
                    href={server.byok_api_key_help_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-info hover:text-info/80 text-sm mt-2 flex items-center gap-1"
                  >
                    Where do I find my API key? <Link2 className="size-3.5" />
                  </a>
                )}
              </div>

              {/* Save toggle */}
              <div className="bg-muted rounded-xl p-4 flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-muted-foreground">
                    <path
                      d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"
                      fill="currentColor"
                    />
                  </svg>
                  <span className="text-sm font-medium text-foreground">Save key for future use</span>
                </div>
                <Switch checked={saveKey} onCheckedChange={setSaveKey} aria-label="Save key for future use" />
              </div>

              {/* Security note */}
              <div className="bg-info/10 rounded-xl p-4 flex items-start gap-3 mb-6">
                <Lock className="mt-0.5 size-4 shrink-0 text-info" />
                <p className="text-sm text-info">
                  Your key is stored securely and transmitted over HTTPS. It is never shared with third parties.
                </p>
              </div>

              <button
                onClick={handleAuthorize}
                disabled={loading}
                className="w-full bg-info hover:bg-info/80 disabled:opacity-60 text-info-foreground font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors"
              >
                <Lock className="size-4" /> Connect &amp; Authorize
              </button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ByokCredentialModal;
