import { describe, expect, it } from "vitest";
import {
  MAX_AUDIO_BYTES,
  MAX_CHAT_ATTACHMENT_BYTES,
  MAX_IMAGE_EDIT_COUNT,
  validateAudioFile,
  validateChatAttachment,
  validateImageEditFile,
} from "./uploadValidation";

function makeFile(name: string, type: string, sizeBytes = 1024): File {
  const content = new Uint8Array(sizeBytes);
  return new File([content], name, { type });
}

describe("validateChatAttachment", () => {
  it("accepts supported image types", () => {
    expect(validateChatAttachment(makeFile("a.png", "image/png"))).toEqual({ ok: true });
    expect(validateChatAttachment(makeFile("a.jpg", "image/jpeg"))).toEqual({ ok: true });
    expect(validateChatAttachment(makeFile("a.webp", "image/webp"))).toEqual({ ok: true });
  });

  it("accepts PDF by mime type or extension", () => {
    expect(validateChatAttachment(makeFile("doc.pdf", "application/pdf"))).toEqual({ ok: true });
    expect(validateChatAttachment(makeFile("doc.PDF", ""))).toEqual({ ok: true });
  });

  it("rejects unsupported types", () => {
    const result = validateChatAttachment(makeFile("notes.txt", "text/plain"));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("not a supported attachment");
    }
  });

  it("rejects files over the size limit", () => {
    const result = validateChatAttachment(makeFile("huge.png", "image/png", MAX_CHAT_ATTACHMENT_BYTES + 1));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("too large");
    }
  });
});

describe("validateImageEditFile", () => {
  it("accepts images under the count limit", () => {
    expect(validateImageEditFile(makeFile("a.png", "image/png"), 0)).toEqual({ ok: true });
  });

  it("rejects when the count limit is reached", () => {
    const result = validateImageEditFile(makeFile("a.png", "image/png"), MAX_IMAGE_EDIT_COUNT);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain(`at most ${MAX_IMAGE_EDIT_COUNT}`);
    }
  });

  it("rejects PDFs for image edits", () => {
    const result = validateImageEditFile(makeFile("doc.pdf", "application/pdf"), 0);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("not a supported image");
    }
  });
});

describe("validateAudioFile", () => {
  it("accepts common audio types", () => {
    expect(validateAudioFile(makeFile("a.mp3", "audio/mpeg"))).toEqual({ ok: true });
    expect(validateAudioFile(makeFile("a.wav", "audio/wav"))).toEqual({ ok: true });
    expect(validateAudioFile(makeFile("a.webm", ""))).toEqual({ ok: true });
  });

  it("rejects non-audio files and oversized files", () => {
    expect(validateAudioFile(makeFile("a.png", "image/png")).ok).toBe(false);
    expect(validateAudioFile(makeFile("a.mp3", "audio/mpeg", MAX_AUDIO_BYTES + 1)).ok).toBe(false);
  });
});
