"use client";

import { useEffect, useMemo } from "react";
import { useSearchParams } from "next/navigation";

const RESULT_STORAGE_KEY = "litellm-mcp-oauth-result";
const RETURN_URL_STORAGE_KEY = "litellm-mcp-oauth-return-url";

const McpOAuthCallbackPage = () => {
  const searchParams = useSearchParams();

  const payload = useMemo(() => {
    if (!searchParams) {
      return null;
    }
    return {
      type: "litellm-mcp-oauth",
      code: searchParams.get("code"),
      state: searchParams.get("state"),
    };
  }, [searchParams]);

  useEffect(() => {
    if (!payload || typeof window === "undefined") {
      return;
    }

    try {
      window.sessionStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(payload));
    } catch (err) {
      console.error("Failed to persist OAuth callback payload", err);
    }

    const returnUrl = window.sessionStorage.getItem(RETURN_URL_STORAGE_KEY);
    console.info("[MCP OAuth callback] returnUrl", returnUrl);
    if (returnUrl) {
      window.location.replace(returnUrl);
    } else {
      window.location.replace("/");
    }
  }, [payload]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
      <div className="max-w-lg w-full rounded-lg bg-white shadow-md p-8 text-center space-y-4">
        <h1 className="text-xl font-semibold text-slate-900">LiteLLM MCP OAuth</h1>
          <p className="text-sm text-slate-700">
            Authorization complete. You may close this window and return to the LiteLLM dashboard.
          </p>
          <p className="text-xs text-slate-500">
            If the window does not close automatically, everything is still saved—you can close it manually.
          </p>
      </div>
    </div>
  );
};

export default McpOAuthCallbackPage;
