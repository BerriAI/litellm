import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIOCRRequest } from "./ocr";

vi.mock("@/components/networking", () => ({
  getGlobalLitellmHeaderName: () => "Authorization",
  getProxyBaseUrl: () => "http://localhost:4000",
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

describe("makeOpenAIOCRRequest", () => {
  const originalFetch = global.fetch;
  const mockUpdateUI = vi.fn();

  beforeEach(() => {
    mockUpdateUI.mockReset();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.clearAllMocks();
  });

  it("posts the file via multipart form to /v1/ocr and renders the joined markdown", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      text: async () =>
        JSON.stringify({
          pages: [
            { index: 0, markdown: "# Hello" },
            { index: 1, markdown: "# World" },
          ],
          model: "mistral/mistral-ocr-latest",
        }),
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const file = new File(["pdf-bytes"], "doc.pdf", { type: "application/pdf" });
    await makeOpenAIOCRRequest({
      file,
      updateUI: mockUpdateUI,
      selectedModel: "mistral/mistral-ocr-latest",
      accessToken: "sk-test",
      tags: ["env:prod"],
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [calledUrl, calledInit] = fetchMock.mock.calls[0];
    expect(calledUrl).toBe("http://localhost:4000/v1/ocr");
    expect(calledInit.method).toBe("POST");

    const headers = calledInit.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer sk-test");
    expect(headers["x-litellm-tags"]).toBe("env:prod");
    expect(headers["Content-Type"]).toBeUndefined();

    const body = calledInit.body as FormData;
    expect(body.get("model")).toBe("mistral/mistral-ocr-latest");
    expect(body.get("file")).toBe(file);

    expect(mockUpdateUI).toHaveBeenCalledWith(
      "## Page 1\n\n# Hello\n\n---\n\n## Page 2\n\n# World",
      "mistral/mistral-ocr-latest",
    );
  });

  it("throws when the proxy returns a non-2xx response", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      text: async () => "model is not configured",
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const file = new File(["pdf"], "doc.pdf", { type: "application/pdf" });

    await expect(
      makeOpenAIOCRRequest({
        file,
        updateUI: mockUpdateUI,
        selectedModel: "mistral/mistral-ocr-latest",
        accessToken: "sk-test",
      }),
    ).rejects.toThrow("model is not configured");
    expect(mockUpdateUI).not.toHaveBeenCalled();
  });
});
