import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import "@/i18n/i18n";
import { testMCPToolsListRequest } from "../components/networking";
import { useTestMCPConnection } from "./useTestMCPConnection";

vi.mock("../components/networking", () => ({ testMCPToolsListRequest: vi.fn() }));

const formValues = {
  url: "https://upstream.example/mcp",
  transport: "http",
  auth_type: "secret_headers",
  static_headers: [{ header: "X-Upstream-Site", value: "example.com" }],
};
const rows = [
  { header: "DD-API-KEY", value: "api-secret" },
  { header: "DD-APPLICATION-KEY", value: "app-secret" },
];

describe("useTestMCPConnection with secret headers", () => {
  beforeEach(() => {
    vi.mocked(testMCPToolsListRequest).mockReset().mockResolvedValue({ tools: [] });
  });

  it("sends the header rows as credentials.headers once every row is complete", async () => {
    const { result } = renderHook(() =>
      useTestMCPConnection({ accessToken: "sk-admin", formValues: { ...formValues, secret_headers: rows } }),
    );
    await waitFor(() => expect(testMCPToolsListRequest).toHaveBeenCalledTimes(1));
    expect(vi.mocked(testMCPToolsListRequest).mock.calls[0][1]).toMatchObject({
      auth_type: "secret_headers",
      static_headers: { "X-Upstream-Site": "example.com" },
      credentials: { headers: { "DD-API-KEY": "api-secret", "DD-APPLICATION-KEY": "app-secret" } },
    });
    expect(result.current.canFetchTools).toBe(true);
  });

  it.each([
    [[]],
    [[{ header: "DD-API-KEY", value: "" }]],
    [
      [
        { header: "DD-API-KEY", value: "api-secret" },
        { header: "", value: "second-secret" },
      ],
    ],
  ])("does not call upstream for incomplete rows %j", (custom) => {
    const { result } = renderHook(() =>
      useTestMCPConnection({ accessToken: "sk-admin", formValues: { ...formValues, secret_headers: custom } }),
    );
    expect(result.current.canFetchTools).toBe(false);
    expect(testMCPToolsListRequest).not.toHaveBeenCalled();
  });
});
