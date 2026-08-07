import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIOcrRequest } from "./OcrApi";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

describe("ocr_api", () => {
  const mockUpdateUI = vi.fn();
  const mockFetch = vi.fn();

  beforeEach(() => {
    const responseBody = JSON.stringify({
      pages: [
        {
          index: 0,
          markdown: "# Extracted text",
        },
      ],
      model: "mistral-ocr-latest",
    });
    mockFetch.mockResolvedValue({
      ok: true,
      text: async () => responseBody,
    } as Response);

    global.fetch = mockFetch;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("posts OCR uploads to /v1/ocr with model and file multipart fields", async () => {
    const file = new File(["document data"], "invoice.pdf", { type: "application/pdf" });

    await makeOpenAIOcrRequest({
      file,
      updateUI: mockUpdateUI,
      selectedModel: "mistral-ocr-latest",
      accessToken: "sk-1234567890",
      tags: ["tag1", "tag2"],
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith("https://example.com/v1/ocr", {
      method: "POST",
      headers: {
        Authorization: "Bearer sk-1234567890",
        "x-litellm-tags": "tag1,tag2",
      },
      body: expect.any(FormData),
      signal: undefined,
    });

    const formData = mockFetch.mock.calls[0][1].body as FormData;
    expect(formData.get("model")).toBe("mistral-ocr-latest");
    expect(formData.get("file")).toBe(file);
    expect(mockUpdateUI).toHaveBeenCalledWith("# Extracted text", "mistral-ocr-latest");
  });
});
