import OpenAI from "openai";
import { v4 as uuid } from "uuid";
import { getGatewayBaseUrl } from "@/components/networking";
import type { ChatImageAttachment } from "./types";

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
  return { id: uuid(), name: file.name, url, mimeType: file.type };
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
    return [{ id: uuid(), name: `Generated image ${index + 1}`, url }];
  });
  if (!images.length) throw new Error("The model did not return an image. Try another prompt or image model.");
  return images;
}
