"use client";

import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import MessageManager from "@/components/molecules/message_manager";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Eye,
  EyeOff,
  Key,
  Link as LinkIcon,
  Lock,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
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
  const [showApiKey, setShowApiKey] = useState(false);
  const [saveKey, setSaveKey] = useState(true);
  const [loading, setLoading] = useState(false);

  const serverDisplayName = server.alias || server.server_name || "Service";
  const firstLetter = serverDisplayName.charAt(0).toUpperCase();

  const handleClose = () => {
    setStep(1);
    setApiKey("");
    setShowApiKey(false);
    setSaveKey(true);
    setLoading(false);
    onClose();
  };

  const handleAuthorize = async () => {
    if (!apiKey.trim()) {
      MessageManager.error("Please enter your API key");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(
        `/v1/mcp/server/${server.server_id}/user-credential`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          body: JSON.stringify({ credential: apiKey.trim(), save: saveKey }),
        },
      );
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err?.detail?.error || "Failed to save credential");
      }
      MessageManager.success(`Connected to ${serverDisplayName}`);
      onSuccess(server.server_id);
      handleClose();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (e: any) {
      MessageManager.error(e.message || "Failed to connect");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-md p-0 overflow-hidden">
        <div className="relative p-6">
          <div className="flex items-center justify-between mb-6">
            {step === 2 ? (
              <button
                onClick={() => setStep(1)}
                className="flex items-center gap-1 text-muted-foreground hover:text-foreground text-sm"
              >
                <ArrowLeft className="h-3.5 w-3.5" /> Back
              </button>
            ) : (
              <div />
            )}
            <div className="flex items-center gap-1.5">
              <div
                className={cn(
                  "w-2 h-2 rounded-full",
                  step === 1 ? "bg-primary" : "bg-muted",
                )}
              />
              <div
                className={cn(
                  "w-2 h-2 rounded-full",
                  step === 2 ? "bg-primary" : "bg-muted",
                )}
              />
            </div>
            <button
              onClick={handleClose}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {step === 1 ? (
            <div className="text-center">
              <div className="flex items-center justify-center gap-3 mb-6">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow">
                  L
                </div>
                <ArrowRight className="text-muted-foreground h-4 w-4" />
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow">
                  {firstLetter}
                </div>
              </div>

              <h2 className="text-2xl font-bold text-foreground mb-2">
                Connect {serverDisplayName}
              </h2>
              <p className="text-muted-foreground mb-6">
                LiteLLM needs access to {serverDisplayName} to complete your
                request.
              </p>

              <div className="bg-muted rounded-xl p-4 text-left mb-4">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5">
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      className="text-muted-foreground"
                    >
                      <rect
                        x="2"
                        y="4"
                        width="20"
                        height="16"
                        rx="2"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                      <path
                        d="M8 4v16M16 4v16"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="font-semibold text-foreground mb-1">
                      How it works
                    </p>
                    <p className="text-muted-foreground text-sm">
                      LiteLLM acts as a secure bridge. Your requests are routed
                      through our MCP client directly to {serverDisplayName}
                      &apos;s API.
                    </p>
                  </div>
                </div>
              </div>

              {server.byok_description &&
                server.byok_description.length > 0 && (
                  <div className="bg-muted rounded-xl p-4 text-left mb-6">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        className="text-emerald-500"
                      >
                        <path
                          d="M12 2L12 22M2 12L22 12"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                        />
                        <circle
                          cx="12"
                          cy="12"
                          r="9"
                          stroke="currentColor"
                          strokeWidth="2"
                        />
                      </svg>
                      Requested Access
                    </p>
                    <ul className="space-y-2">
                      {server.byok_description.map((item, i) => (
                        <li
                          key={i}
                          className="flex items-center gap-2 text-sm text-foreground"
                        >
                          <Check className="text-emerald-500 flex-shrink-0 h-4 w-4" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

              <Button
                onClick={() => setStep(2)}
                className="w-full"
              >
                Continue to Authentication <ArrowRight className="h-4 w-4" />
              </Button>
              <Button
                onClick={handleClose}
                variant="ghost"
                className="mt-3 w-full text-muted-foreground"
              >
                Cancel
              </Button>
            </div>
          ) : (
            <div>
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                <Key className="text-primary h-5 w-5" />
              </div>

              <h2 className="text-2xl font-bold text-foreground mb-2">
                Provide API Key
              </h2>
              <p className="text-muted-foreground mb-6">
                Enter your {serverDisplayName} API key to authorize this
                connection.
              </p>

              <div className="mb-4 space-y-2">
                <Label htmlFor="byok-api-key">
                  {serverDisplayName} API Key
                </Label>
                <div className="relative">
                  <Input
                    id="byok-api-key"
                    type={showApiKey ? "text" : "password"}
                    placeholder="Enter your API key"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground"
                    aria-label={showApiKey ? "Hide api key" : "Show api key"}
                  >
                    {showApiKey ? (
                      <EyeOff size={14} />
                    ) : (
                      <Eye size={14} />
                    )}
                  </button>
                </div>
                {server.byok_api_key_help_url && (
                  <a
                    href={server.byok_api_key_help_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline text-sm mt-2 flex items-center gap-1"
                  >
                    Where do I find my API key?{" "}
                    <LinkIcon className="h-3.5 w-3.5" />
                  </a>
                )}
              </div>

              <div className="bg-muted rounded-xl p-4 flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    className="text-muted-foreground"
                  >
                    <path
                      d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"
                      fill="currentColor"
                    />
                  </svg>
                  <span className="text-sm font-medium text-foreground">
                    Save key for future use
                  </span>
                </div>
                <Switch
                  checked={saveKey}
                  onCheckedChange={setSaveKey}
                />
              </div>

              <div className="bg-primary/5 rounded-xl p-4 flex items-start gap-3 mb-6">
                <Lock className="text-primary mt-0.5 flex-shrink-0 h-4 w-4" />
                <p className="text-sm text-primary">
                  Your key is stored securely and transmitted over HTTPS. It is
                  never shared with third parties.
                </p>
              </div>

              <Button
                onClick={handleAuthorize}
                disabled={loading}
                className="w-full"
              >
                <Lock className="h-4 w-4" /> Connect &amp; Authorize
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ByokCredentialModal;
