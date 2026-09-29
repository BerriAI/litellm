import { existsSync, createReadStream } from 'fs';
import { basename, extname } from 'path';
import type { ToolContext } from '../types.js';

const MIME_BY_EXT: { [ext: string]: string } = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
};

export interface UploadedImage {
  fileId: string;
  webContentLink: string;
}

export async function uploadImageToDrive(
  ctx: ToolContext,
  localFilePath: string,
  options: { parentFolderId?: string; makePublic?: boolean } = {},
): Promise<UploadedImage> {
  const { parentFolderId, makePublic = false } = options;

  if (!existsSync(localFilePath)) {
    throw new Error(`Image file not found: ${localFilePath}`);
  }

  const fileName = basename(localFilePath);
  const ext = extname(localFilePath).toLowerCase();
  const mimeType = MIME_BY_EXT[ext] || 'application/octet-stream';

  const requestBody: { name: string; mimeType: string; parents?: string[] } = {
    name: fileName,
    mimeType,
  };
  if (parentFolderId) requestBody.parents = [parentFolderId];

  const drive = ctx.getDrive();

  const uploadResponse = await drive.files.create({
    requestBody,
    media: { mimeType, body: createReadStream(localFilePath) },
    fields: 'id,webViewLink,webContentLink',
    supportsAllDrives: true,
  });

  const fileId = uploadResponse.data.id;
  if (!fileId) throw new Error('Failed to upload image to Drive - no file ID returned');

  if (makePublic) {
    await drive.permissions.create({
      fileId,
      requestBody: { role: 'reader', type: 'anyone' },
    });
  }

  const fileInfo = await drive.files.get({
    fileId,
    fields: 'webContentLink',
    supportsAllDrives: true,
  });

  const webContentLink = fileInfo.data.webContentLink;
  if (!webContentLink) throw new Error('Failed to get web content link for uploaded image');

  return { fileId, webContentLink };
}

export async function deleteDriveFile(ctx: ToolContext, fileId: string): Promise<void> {
  await ctx.getDrive().files.delete({ fileId, supportsAllDrives: true });
}
