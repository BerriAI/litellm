import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MCPConnect from "./mcp_connect";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://runtime.litellm.test"),
}));

vi.mock("@/utils/dataUtils", () => ({
  copyToClipboard: vi.fn(),
}));

describe("MCPConnect", () => {
  it("renders connection snippets with the API docs base url when configured", () => {
    const apiDocBaseUrl = "https://docs.litellm.test";
    const proxyBaseUrl = "https://proxy.litellm.test";

    const { container } = render(
      <MCPConnect
        proxySettings={{
          LITELLM_UI_API_DOC_BASE_URL: apiDocBaseUrl,
          PROXY_BASE_URL: proxyBaseUrl,
        }}
      />,
    );

    expect(container.textContent).toContain(`${apiDocBaseUrl}/mcp`);
    expect(container.textContent).toContain(`${apiDocBaseUrl}/v1/responses`);
    expect(container.textContent).not.toContain(`${proxyBaseUrl}/mcp`);
  });
});
