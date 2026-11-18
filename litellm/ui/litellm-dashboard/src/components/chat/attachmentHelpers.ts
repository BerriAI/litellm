import type { MultimodalContent } from "@/components/chat_ui/types";
import type { ResponsesMessage } from "@/components/llm_calls/responses_api";
import { v4 as uuid } from "uuid";
import { CHAT_IMAGE_ACCEPT, MAX_CHAT_IMAGE_BYTES, isChatImageUrl, readChatImage } from "./imageHelpers";
import type { ChatAttachment, ChatFileAttachment, ChatMessage } from "./types";

export const MAX_CHAT_ATTACHMENTS = 4;
const DOCUMENT_TYPES: Record<string, string> = {
  pdf: "application/pdf",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  csv: "text/csv",
  tsv: "text/tab-separated-values",
  txt: "text/plain",
  md: "text/markdown",
  json: "application/json",
  html: "text/html",
  xml: "application/xml",
};
const CODE_EXTENSIONS = [
  "c",
  "cpp",
  "css",
  "go",
  "h",
  "java",
  "js",
  "jsx",
  "log",
  "py",
  "rb",
  "rs",
  "sh",
  "sql",
  "ts",
  "tsx",
  "yaml",
  "yml",
];
for (const extension of CODE_EXTENSIONS) DOCUMENT_TYPES[extension] = "text/plain";

export const CHAT_ATTACHMENT_ACCEPT = [
  CHAT_IMAGE_ACCEPT,
  ...Object.keys(DOCUMENT_TYPES).map((extension) => `.${extension}`),
].join(",");

export function isChatFileData(file: ChatFileAttachment): boolean {
  return Boolean(
    Object.values(DOCUMENT_TYPES).includes(file.mimeType) &&
      file.url.startsWith(`data:${file.mimeType};base64,`) &&
      /^[A-Za-z0-9+/]+={0,2}$/.test(file.url.split(",")[1] ?? ""),
  );
}

export async function readChatAttachment(file: File): Promise<ChatAttachment> {
  if (file.type.startsWith("image/")) return readChatImage(file);
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  const mimeType = Object.hasOwn(DOCUMENT_TYPES, extension) ? DOCUMENT_TYPES[extension] : undefined;
  if (!mimeType) throw new Error("Choose an image, PDF, Word (.docx), Excel (.xlsx), text, or code file.");
  if (file.size === 0) throw new Error(`${file.name} is empty. Choose another file.`);
  if (file.size > MAX_CHAT_IMAGE_BYTES) throw new Error(`${file.name} is too large. Files must be 10 MB or smaller.`);
  const url = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error(`Could not read ${file.name}. Choose the file again.`));
    };
    reader.onerror = () => reject(new Error(`Could not read ${file.name}. Choose the file again.`));
    reader.onabort = () => reject(new Error(`Reading ${file.name} was canceled.`));
    reader.readAsDataURL(new Blob([file], { type: mimeType }));
  });
  return { id: uuid(), kind: "file", name: file.name, url, mimeType };
}

export function toResponsesMessages(
  messages: Pick<
    ChatMessage,
    "role" | "content" | "attachments" | "generatedImages" | "imageGeneration" | "mcpEvents"
  >[],
): ResponsesMessage[] {
  return messages.flatMap((message): ResponsesMessage[] => {
    if (message.role === "tool") return [];
    if (message.role === "assistant") {
      return message.content.trim() || message.mcpEvents?.length
        ? [
            {
              role: "assistant",
              content: message.content,
              ...(message.mcpEvents?.length ? { mcpEvents: message.mcpEvents } : {}),
            },
          ]
        : [];
    }
    if (!message.attachments?.length) return [{ role: "user", content: message.content }];
    const content: MultimodalContent[] = [];
    if (message.content.trim()) content.push({ type: "input_text", text: message.content });
    for (const attachment of message.attachments) {
      if (attachment.kind === "file") {
        if (!isChatFileData(attachment)) {
          throw new Error(`A saved file (${attachment.name}) is unavailable. Start a new chat and attach it again.`);
        }
        content.push({ type: "input_file", filename: attachment.name, file_data: attachment.url });
        continue;
      }
      if (!isChatImageUrl(attachment.url)) {
        throw new Error(`A saved image (${attachment.name}) is unavailable. Start a new chat and attach it again.`);
      }
      content.push({ type: "input_image", image_url: attachment.url, detail: "auto" });
    }
    return [{ role: "user", content }];
  });
}
