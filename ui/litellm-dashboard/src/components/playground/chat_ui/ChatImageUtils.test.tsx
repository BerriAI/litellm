import { describe, expect, it, vi, beforeEach } from "vitest";
import {
  convertImageToBase64,
  createChatMultimodalMessage,
  createChatDisplayMessage,
  shouldShowChatAttachedImage,
} from "./ChatImageUtils";
import { MessageType } from "./types";

describe("ChatImageUtils", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("convertImageToBase64", () => {
    it("should convert file to base64 data URI", async () => {
      const file = new File(["test content"], "test.png", { type: "image/png" });
      const result = await convertImageToBase64(file);
      expect(result).toMatch(/^data:image\/png;base64,/);
    });

    it("should handle different file types", async () => {
      const jpegFile = new File(["jpeg content"], "test.jpg", { type: "image/jpeg" });
      const result = await convertImageToBase64(jpegFile);
      expect(result).toMatch(/^data:image\/jpeg;base64,/);
    });

    it("should reject on file read error", async () => {
      const file = new File(["test"], "test.png", { type: "image/png" });
      const originalReadAsDataURL = FileReader.prototype.readAsDataURL;

      FileReader.prototype.readAsDataURL = vi.fn(function (this: FileReader) {
        setTimeout(() => {
          if (this.onerror) {
            this.onerror(new Error("Read error") as any);
          }
        }, 0);
      });

      await expect(convertImageToBase64(file)).rejects.toThrow();

      FileReader.prototype.readAsDataURL = originalReadAsDataURL;
    });
  });

  describe("createChatMultimodalMessage", () => {
    it("should create multimodal message with text and image", async () => {
      const file = new File(["test content"], "test.png", { type: "image/png" });
      const inputMessage = "What is in this image?";

      const result = await createChatMultimodalMessage(inputMessage, file);

      expect(result.role).toBe("user");
      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toEqual({ type: "text", text: inputMessage });
      expect(result.content[1]).toMatchObject({
        type: "image_url",
        image_url: {
          url: expect.stringMatching(/^data:image\/png;base64,/),
        },
      });
    });

    it("should include base64 data URI in image_url", async () => {
      const file = new File(["test content"], "test.png", { type: "image/png" });
      const result = await createChatMultimodalMessage("test", file);

      const imageContent = result.content[1];
      expect(imageContent.type).toBe("image_url");
      if ("image_url" in imageContent && imageContent.image_url) {
        expect(imageContent.image_url.url).toMatch(/^data:/);
      }
    });
  });

  describe("createChatDisplayMessage", () => {
    it("should create display message without file", () => {
      const result = createChatDisplayMessage("Hello world", false);

      expect(result.role).toBe("user");
      expect(result.content).toBe("Hello world");
      expect(result.imagePreviewUrl).toBeUndefined();
    });

    it("should create display message with PDF file", () => {
      const filePreviewUrl = "blob:test-url";
      const result = createChatDisplayMessage("Read this", true, filePreviewUrl, "document.pdf");

      expect(result.content).toBe("Read this [PDF attached]");
      expect(result.imagePreviewUrl).toBe(filePreviewUrl);
    });

    it("should create display message with image file", () => {
      const filePreviewUrl = "blob:test-url";
      const result = createChatDisplayMessage("Look at this", true, filePreviewUrl, "photo.jpg");

      expect(result.content).toBe("Look at this [Image attached]");
      expect(result.imagePreviewUrl).toBe(filePreviewUrl);
    });

    it("should create display message with file but no fileName", () => {
      const filePreviewUrl = "blob:test-url";
      const result = createChatDisplayMessage("Check this", true, filePreviewUrl);

      expect(result.content).toBe("Check this ");
      expect(result.imagePreviewUrl).toBe(filePreviewUrl);
    });

    it("should create display message with file but no preview URL", () => {
      const result = createChatDisplayMessage("See this", true, undefined, "image.png");

      expect(result.content).toBe("See this [Image attached]");
      expect(result.imagePreviewUrl).toBeUndefined();
    });
  });

  describe("shouldShowChatAttachedImage", () => {
    it("should return true for user message with image attachment", () => {
      const message: MessageType = {
        role: "user",
        content: "Check this [Image attached]",
        imagePreviewUrl: "blob:test-url",
      };

      expect(shouldShowChatAttachedImage(message)).toBe(true);
    });

    it("should return true for user message with PDF attachment", () => {
      const message: MessageType = {
        role: "user",
        content: "Read this [PDF attached]",
        imagePreviewUrl: "blob:test-url",
      };

      expect(shouldShowChatAttachedImage(message)).toBe(true);
    });

    it("should return false for assistant message", () => {
      const message: MessageType = {
        role: "assistant",
        content: "Here is the image [Image attached]",
        imagePreviewUrl: "blob:test-url",
      };

      expect(shouldShowChatAttachedImage(message)).toBe(false);
    });

    it("should return false when content is not a string", () => {
      const message: MessageType = {
        role: "user",
        content: [{ type: "input_text", text: "test" }],
        imagePreviewUrl: "blob:test-url",
      };

      expect(shouldShowChatAttachedImage(message)).toBe(false);
    });

    it("should return false when content does not include attachment marker", () => {
      const message: MessageType = {
        role: "user",
        content: "Just regular text",
        imagePreviewUrl: "blob:test-url",
      };

      expect(shouldShowChatAttachedImage(message)).toBe(false);
    });

    it("should return false when imagePreviewUrl is missing", () => {
      const message: MessageType = {
        role: "user",
        content: "Check this [Image attached]",
      };

      expect(shouldShowChatAttachedImage(message)).toBe(false);
    });

    it("should return false when imagePreviewUrl is empty string", () => {
      const message: MessageType = {
        role: "user",
        content: "Check this [Image attached]",
        imagePreviewUrl: "",
      };

      expect(shouldShowChatAttachedImage(message)).toBe(false);
    });
  });
});
