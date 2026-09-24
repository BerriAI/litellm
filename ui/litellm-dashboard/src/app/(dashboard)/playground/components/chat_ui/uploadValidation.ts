export type UploadValidationResult = { ok: true } | { ok: false; error: string };

export const CHAT_ATTACHMENT_ACCEPT = "image/png,image/jpeg,image/jpg,image/gif,image/webp,application/pdf,.pdf";
export const IMAGE_EDIT_ACCEPT = "image/png,image/jpeg,image/jpg,image/gif,image/webp";
export const AUDIO_ACCEPT = "audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm";

export const MAX_CHAT_ATTACHMENT_BYTES = 20 * 1024 * 1024;
export const MAX_IMAGE_EDIT_BYTES = 20 * 1024 * 1024;
export const MAX_AUDIO_BYTES = 25 * 1024 * 1024;
export const MAX_IMAGE_EDIT_COUNT = 10;
export const MAX_CHAT_ATTACHMENT_COUNT = 1;

const IMAGE_MIME_TYPES = new Set(["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]);
const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp"]);
const PDF_MIME_TYPES = new Set(["application/pdf"]);
const PDF_EXTENSIONS = new Set([".pdf"]);
const AUDIO_MIME_PREFIX = "audio/";
const AUDIO_EXTENSIONS = new Set([".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]);

function getExtension(fileName: string): string {
  const dot = fileName.lastIndexOf(".");
  if (dot < 0) {
    return "";
  }
  return fileName.slice(dot).toLowerCase();
}

function formatMb(bytes: number): string {
  return `${Math.round(bytes / (1024 * 1024))} MB`;
}

function isImageFile(file: File): boolean {
  if (IMAGE_MIME_TYPES.has(file.type)) {
    return true;
  }
  return IMAGE_EXTENSIONS.has(getExtension(file.name));
}

function isPdfFile(file: File): boolean {
  if (PDF_MIME_TYPES.has(file.type)) {
    return true;
  }
  return PDF_EXTENSIONS.has(getExtension(file.name));
}

function isAudioFile(file: File): boolean {
  if (file.type.startsWith(AUDIO_MIME_PREFIX)) {
    return true;
  }
  return AUDIO_EXTENSIONS.has(getExtension(file.name));
}

function validateSize(file: File, maxBytes: number): UploadValidationResult {
  if (file.size <= maxBytes) {
    return { ok: true };
  }
  return {
    ok: false,
    error: `"${file.name}" is too large. Maximum size is ${formatMb(maxBytes)}.`,
  };
}

export function validateChatAttachment(file: File): UploadValidationResult {
  if (!isImageFile(file) && !isPdfFile(file)) {
    return {
      ok: false,
      error: `"${file.name}" is not a supported attachment. Use PNG, JPEG, GIF, WebP, or PDF.`,
    };
  }
  return validateSize(file, MAX_CHAT_ATTACHMENT_BYTES);
}

export function validateImageEditFile(file: File, currentCount: number): UploadValidationResult {
  if (currentCount >= MAX_IMAGE_EDIT_COUNT) {
    return {
      ok: false,
      error: `You can upload at most ${MAX_IMAGE_EDIT_COUNT} images.`,
    };
  }
  if (!isImageFile(file)) {
    return {
      ok: false,
      error: `"${file.name}" is not a supported image. Use PNG, JPEG, GIF, or WebP.`,
    };
  }
  return validateSize(file, MAX_IMAGE_EDIT_BYTES);
}

export function validateAudioFile(file: File): UploadValidationResult {
  if (!isAudioFile(file)) {
    return {
      ok: false,
      error: `"${file.name}" is not a supported audio file. Use MP3, MP4, MPEG, MPGA, M4A, WAV, or WEBM.`,
    };
  }
  return validateSize(file, MAX_AUDIO_BYTES);
}
