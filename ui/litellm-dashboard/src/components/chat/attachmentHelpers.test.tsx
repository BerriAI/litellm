import { describe, expect, it } from "vitest";
import { readChatAttachment, toResponsesMessages } from "./attachmentHelpers";
import { MAX_CHAT_IMAGE_BYTES } from "./imageHelpers";
import type { ChatFileAttachment, MCPEvent } from "./types";

const file: ChatFileAttachment = {
  id: "document-1",
  kind: "file",
  name: "brief.pdf",
  mimeType: "application/pdf",
  url: "data:application/pdf;base64,cGRm",
};

describe("chat attachments", () => {
  it.each([
    ["brief.pdf", "application/pdf"],
    ["brief.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    ["orders.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
    ["notes.txt", "text/plain"],
    ["example.py", "text/plain"],
  ])("should read %s with its canonical MIME type even when the browser omits it", async (name, mimeType) => {
    const attachment = await readChatAttachment(new File(["fixture"], name));
    const expected = { kind: "file", name, mimeType, url: `data:${mimeType};base64,Zml4dHVyZQ==` };
    expect(attachment).toMatchObject(expected);
  });

  it("should reject unsupported, empty, and oversized documents", async () => {
    await expect(readChatAttachment(new File(["binary"], "installer.exe"))).rejects.toThrow("Choose an image, PDF");
    await expect(readChatAttachment(new File(["binary"], "file.constructor"))).rejects.toThrow("Choose an image, PDF");
    await expect(readChatAttachment(new File(["legacy"], "legacy.xls"))).rejects.toThrow("Excel (.xlsx)");
    await expect(readChatAttachment(new File([], "empty.pdf"))).rejects.toThrow("empty");
    const large = new File(["pdf"], "large.pdf");
    Object.defineProperty(large, "size", { value: MAX_CHAT_IMAGE_BYTES + 1 });
    await expect(readChatAttachment(large)).rejects.toThrow("10 MB");
  });

  it("should preserve mixed files, images, and tool-only assistant history for Responses", () => {
    const image = { id: "image-1", name: "diagram.png", url: "data:image/png;base64,aW1hZ2U=" };
    const mcpEvents: MCPEvent[] = [
      {
        type: "response.output_item.done",
        item: { type: "mcp_call", id: "lookup-1", name: "lookup", arguments: "{}", output: "Found it" },
      },
    ];
    expect(
      toResponsesMessages([
        { role: "user", content: "Compare", attachments: [image, file] },
        { role: "assistant", content: "I read the file.", mcpEvents },
        { role: "assistant", content: "", mcpEvents },
      ]),
    ).toEqual([
      {
        role: "user",
        content: [
          { type: "input_text", text: "Compare" },
          { type: "input_image", image_url: image.url, detail: "auto" },
          { type: "input_file", filename: file.name, file_data: file.url },
        ],
      },
      { role: "assistant", content: "I read the file.", mcpEvents },
      { role: "assistant", content: "", mcpEvents },
    ]);
  });

  it("should fail clearly instead of silently dropping unavailable document contents", () => {
    for (const url of ["", "javascript:alert(1)", "https://example.com/private.pdf", "data:text/html;base64,cGRm"]) {
      expect(() => toResponsesMessages([{ role: "user", content: "Read", attachments: [{ ...file, url }] }])).toThrow(
        "A saved file (brief.pdf) is unavailable",
      );
    }
  });
});
