import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import MCPConnect from "./mcp_connect";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "http://proxy.internal:4000",
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test" }),
}));

vi.mock("@/app/(dashboard)/hooks/proxySettings/useProxySettings", () => ({
  default: vi.fn(),
}));

const mockedUseProxySettings = vi.mocked(useProxySettings);

const proxySettings = (docBaseUrl: string | null) => ({
  PROXY_BASE_URL: "http://proxy.internal:4000",
  PROXY_LOGOUT_URL: "",
  LITELLM_UI_API_DOC_BASE_URL: docBaseUrl,
});

describe("MCPConnect", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the Server URL from LITELLM_UI_API_DOC_BASE_URL when it is set", () => {
    mockedUseProxySettings.mockReturnValue(proxySettings("https://gateway.public.example.com"));

    render(<MCPConnect />);

    expect(screen.getAllByText("https://gateway.public.example.com/mcp").length).toBeGreaterThan(0);
    expect(screen.queryAllByText(/proxy\.internal/)).toHaveLength(0);
  });

  it("renders the Server URL from the proxy base url when LITELLM_UI_API_DOC_BASE_URL is unset", () => {
    mockedUseProxySettings.mockReturnValue(proxySettings(null));

    render(<MCPConnect />);

    expect(screen.getAllByText("http://proxy.internal:4000/mcp").length).toBeGreaterThan(0);
  });
});
