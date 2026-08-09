import { describe, expect, it } from "vitest";
import { resolveMCPConnectBaseUrl } from "./mcpConnectBaseUrl";

describe("resolveMCPConnectBaseUrl", () => {
  const fallbackBaseUrl = "https://runtime.litellm.test";

  it("uses LITELLM_UI_API_DOC_BASE_URL when provided", () => {
    expect(
      resolveMCPConnectBaseUrl(
        {
          LITELLM_UI_API_DOC_BASE_URL: "https://docs.litellm.test",
        },
        fallbackBaseUrl,
      ),
    ).toBe("https://docs.litellm.test");
  });

  it("prefers LITELLM_UI_API_DOC_BASE_URL over PROXY_BASE_URL", () => {
    expect(
      resolveMCPConnectBaseUrl(
        {
          LITELLM_UI_API_DOC_BASE_URL: "https://docs.litellm.test",
          PROXY_BASE_URL: "https://proxy.litellm.test",
        },
        fallbackBaseUrl,
      ),
    ).toBe("https://docs.litellm.test");
  });

  it("falls back to PROXY_BASE_URL when the docs url is missing", () => {
    expect(
      resolveMCPConnectBaseUrl(
        {
          PROXY_BASE_URL: "https://proxy.litellm.test",
        },
        fallbackBaseUrl,
      ),
    ).toBe("https://proxy.litellm.test");
  });

  it("falls back to the runtime base url when settings are empty", () => {
    expect(
      resolveMCPConnectBaseUrl(
        {
          LITELLM_UI_API_DOC_BASE_URL: "   ",
          PROXY_BASE_URL: "",
        },
        fallbackBaseUrl,
      ),
    ).toBe(fallbackBaseUrl);
  });

  it("trims trailing slashes before endpoint paths are appended", () => {
    expect(
      resolveMCPConnectBaseUrl(
        {
          LITELLM_UI_API_DOC_BASE_URL: " https://docs.litellm.test/// ",
        },
        fallbackBaseUrl,
      ),
    ).toBe("https://docs.litellm.test");

    expect(
      resolveMCPConnectBaseUrl(
        {
          PROXY_BASE_URL: "https://proxy.litellm.test/",
        },
        `${fallbackBaseUrl}/`,
      ),
    ).toBe("https://proxy.litellm.test");

    expect(resolveMCPConnectBaseUrl(undefined, `${fallbackBaseUrl}/`)).toBe(fallbackBaseUrl);
  });
});
