import { toResponsesMessages } from "./attachmentHelpers";
import { beforeEach, describe, expect, it, vi } from "vitest";
import OpenAI from "openai";
import { generateChatImages, isChatImageUrl, MAX_CHAT_IMAGE_BYTES, readChatImage } from "./imageHelpers";

const { generate } = vi.hoisted(() => ({ generate: vi.fn() }));
vi.mock("openai", () => ({
  default: vi.fn(function () {
    return { images: { generate } };
  }),
}));
vi.mock("@/components/networking", () => ({ getGatewayBaseUrl: () => "https://gateway.example/v1" }));

const image = { id: "image-1", name: "design.png", url: "data:image/png;base64,aW1hZ2U=" };

describe("chat image helpers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should read supported images for preview and requests", async () => {
    const result = await readChatImage(new File(["image"], "design.png", { type: "image/png" }));
    const expected = { id: expect.any(String), name: "design.png", url: image.url, mimeType: "image/png" };
    expect(result).toEqual(expected);
  });

  it("should reject unsupported, empty, and oversized uploads", async () => {
    await expect(readChatImage(new File(["pdf"], "design.pdf", { type: "application/pdf" }))).rejects.toThrow(
      "PNG, JPEG",
    );
    await expect(readChatImage(new File([], "empty.png", { type: "image/png" }))).rejects.toThrow("empty");
    const large = new File(["image"], "large.png", { type: "image/png" });
    Object.defineProperty(large, "size", { value: MAX_CHAT_IMAGE_BYTES + 1 });
    await expect(readChatImage(large)).rejects.toThrow("10 MB");
  });

  it("should replay user image attachments using Responses API input parts", () => {
    expect(
      toResponsesMessages([
        { role: "user", content: "Describe this", attachments: [image] },
        { role: "assistant", content: "A design" },
        { role: "user", content: "", attachments: [image] },
        { role: "assistant", content: "", generatedImages: [image] },
        { role: "tool", content: "tool output" },
      ]),
    ).toEqual([
      {
        role: "user",
        content: [
          { type: "input_text", text: "Describe this" },
          { type: "input_image", image_url: image.url, detail: "auto" },
        ],
      },
      { role: "assistant", content: "A design" },
      { role: "user", content: [{ type: "input_image", image_url: image.url, detail: "auto" }] },
    ]);
  });

  it("should stop a replay if an image could not be restored", () => {
    expect(() =>
      toResponsesMessages([{ role: "user", content: "Look", attachments: [{ ...image, url: "" }] }]),
    ).toThrow("Start a new chat and attach it again");
  });

  it("should reject executable and non-image URLs", () => {
    expect(isChatImageUrl("javascript:alert(1)")).toBe(false);
    expect(isChatImageUrl("data:text/html;base64,aW1hZ2U=")).toBe(false);
    expect(isChatImageUrl("data:image/svg+xml;base64,aW1hZ2U=")).toBe(false);
    expect(isChatImageUrl("https://images.example/generated.png")).toBe(true);
  });

  it("should generate through the configured gateway and accept URLs and base64 results", async () => {
    const signal = new AbortController().signal;
    generate.mockResolvedValue({ data: [{ url: "https://images.example/generated.png" }, { b64_json: "aW1hZ2U=" }] });
    const request = {
      prompt: "A small boat",
      model: "company-image-model",
      accessToken: "test-token",
      signal,
    };
    const result = await generateChatImages(request);
    expect(OpenAI).toHaveBeenCalledWith({
      apiKey: "test-token",
      baseURL: "https://gateway.example/v1",
      dangerouslyAllowBrowser: true,
    });
    expect(generate).toHaveBeenCalledWith({ model: "company-image-model", prompt: "A small boat" }, { signal });
    expect(result.map((entry) => entry.url)).toEqual(["https://images.example/generated.png", image.url]);
  });

  it("should surface generation failures and aborts to the caller", async () => {
    generate.mockResolvedValue({ data: [] });
    await expect(generateChatImages({ prompt: "Boat", model: "image-model", accessToken: "token" })).rejects.toThrow(
      "did not return an image",
    );
    const aborted = new DOMException("Stopped", "AbortError");
    generate.mockRejectedValue(aborted);
    await expect(generateChatImages({ prompt: "Boat", model: "image-model", accessToken: "token" })).rejects.toBe(
      aborted,
    );
  });

  it("should preserve the generated image format returned by the provider", async () => {
    generate.mockResolvedValue({ output_format: "webp", data: [{ b64_json: "aW1hZ2U=" }] });
    const images = await generateChatImages({ prompt: "Boat", model: "image-model", accessToken: "token" });
    expect(images[0].url).toBe("data:image/webp;base64,aW1hZ2U=");
  });
});
