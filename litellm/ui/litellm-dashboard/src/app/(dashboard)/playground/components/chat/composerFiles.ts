import type { MessageAttachment } from "@/components/chat_ui/types";
import { validateAudioFile, validateChatAttachment, validateImageEditFile } from "../chat_ui/uploadValidation";
import type { AttachKind } from "./endpointCapabilities";

export type DraftFile = Readonly<{ file: File; previewUrl?: string }>;

export type AddFilesResult = Readonly<{ files: readonly DraftFile[]; errors: readonly string[] }>;

const isImage = (file: File): boolean => file.type.startsWith("image/");

const kindOf = (file: File): MessageAttachment["kind"] => {
  if (file.type.startsWith("audio/")) return "audio";
  if (isImage(file)) return "image";
  return "pdf";
};

export const toAttachment = ({ file, previewUrl }: DraftFile): MessageAttachment => ({
  name: file.name,
  size: file.size,
  kind: kindOf(file),
  ...(previewUrl ? { previewUrl } : {}),
});

const validate = (kind: AttachKind, file: File, count: number) => {
  if (kind === "images") return validateImageEditFile(file, count);
  if (kind === "audio") return validateAudioFile(file);
  return validateChatAttachment(file);
};

export const addFiles = (
  kind: AttachKind,
  current: readonly DraftFile[],
  incoming: readonly File[],
  preview: (file: File) => string | undefined,
): AddFilesResult => {
  const single = kind !== "images";
  return incoming.reduce<AddFilesResult>(
    (result, file) => {
      const kept = single ? [] : result.files;
      const check = validate(kind, file, kept.length);
      if (!check.ok) return { ...result, errors: [...result.errors, check.error] };
      return { ...result, files: [...kept, { file, previewUrl: isImage(file) ? preview(file) : undefined }] };
    },
    { files: current, errors: [] },
  );
};

export const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};
