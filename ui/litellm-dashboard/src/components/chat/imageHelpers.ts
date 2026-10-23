import OpenAI from "openai";
import { getGatewayBaseUrl } from "@/components/networking";
import type { MessageType, MultimodalContent } from "@/components/chat_ui/types";
import type { ChatImageAttachment, ChatMessage } from "./types";

export const MAX_CHAT_IMAGES = 4;
export const MAX_CHAT_IMAGE_BYTES = 10 * 1024 * 1024;
export const CHAT_IMAGE_ACCEPT = "image/png,image/jpeg,image/webp,image/gif";
const IMAGE_TYPES = new Set(CHAT_IMAGE_ACCEPT.split(","));
const DATA_IMAGE_PATTERN = /^data:image\/(png|jpeg|webp|gif);base64,[a-zA-Z0-9+/]/;

export function isChatImageUrl(url: string): boolean {
  if (DATA_IMAGE_PATTERN.test(url)) return true;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:";
  } catch {
    return false;
  }
}

export async function readChatImage(file: File): Promise<ChatImageAttachment> {
  if (!IMAGE_TYPES.has(file.type)) throw new Error("Choose a PNG, JPEG, WebP, or GIF image.");
  if (file.size === 0) throw new Error(`${file.name} is empty. Choose another image.`);
  if (file.size > MAX_CHAT_IMAGE_BYTES) {
    throw new Error(`${file.name} is too large. Images must be 10 MB or smaller.`);
  }
  const url = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string" && DATA_IMAGE_PATTERN.test(reader.result)) resolve(reader.result);
      else reject(new Error(`Could not read ${file.name}. Choose the image again.`));
    };
    reader.onerror = () => reject(new Error(`Could not read ${file.name}. Choose the image again.`));
    reader.onabort = () => reject(new Error(`Reading ${file.name} was canceled.`));
    reader.readAsDataURL(file);
  });
  return { id: crypto.randomUUID(), name: file.name, url, mimeType: file.type };
}

export function toResponsesMessages(
  messages: Pick<ChatMessage, "role" | "content" | "attachments" | "generatedImages" | "imageGeneration">[],
): MessageType[] {
  return messages.flatMap((message): MessageType[] => {
    if (message.role === "tool") return [];
    if (message.role === "assistant") {
      return message.content.trim() ? [{ role: "assistant", content: message.content }] : [];
    }
    if (!message.attachments?.length) return [{ role: "user", content: message.content }];
    const content: MultimodalContent[] = [];
    if (message.content.trim()) content.push({ type: "input_text", text: message.content });
    for (const attachment of message.attachments) {
      if (!isChatImageUrl(attachment.url)) {
        throw new Error(`A saved image (${attachment.name}) is unavailable. Start a new chat and attach it again.`);
      }
      content.push({ type: "input_image", image_url: attachment.url, detail: "auto" });
    }
    return [{ role: "user", content }];
  });
}

export async function generateChatImages({
  prompt,
  model,
  accessToken,
  signal,
}: {
  prompt: string;
  model: string;
  accessToken: string;
  signal?: AbortSignal;
}): Promise<ChatImageAttachment[]> {
  const client = new OpenAI({ apiKey: accessToken, baseURL: getGatewayBaseUrl(), dangerouslyAllowBrowser: true });
  const response = await client.images.generate({ model, prompt }, { signal });
  const mimeType = `image/${response.output_format ?? "png"}`;
  const images = (response.data ?? []).flatMap((image, index): ChatImageAttachment[] => {
    const url = image.b64_json ? `data:${mimeType};base64,${image.b64_json}` : image.url;
    if (!url || !isChatImageUrl(url)) return [];
    return [{ id: crypto.randomUUID(), name: `Generated image ${index + 1}`, url }];
  });
  if (!images.length) throw new Error("The model did not return an image. Try another prompt or image model.");
  return images;
}
