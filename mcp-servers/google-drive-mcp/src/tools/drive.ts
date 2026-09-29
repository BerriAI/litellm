import { z } from 'zod';
import type { drive_v3 } from 'googleapis';
import { existsSync, statSync, createReadStream } from 'fs';
import { Readable } from 'stream';
import { mkdtemp, readFile, writeFile, rm } from 'fs/promises';
import { tmpdir } from 'os';
import { basename, extname, join } from 'path';
import pdfLib from 'pdf-lib';
const { PDFDocument } = pdfLib;
import type { ToolDefinition, ToolContext, ToolResult } from '../types.js';
import { errorResponse } from '../types.js';
import { escapeDriveQuery, getMimeTypeFromFilename, isTextMime, TEXT_MIME_TYPES, ALL_DRIVES_LIST_PARAMS, PARENT_SCOPED_LIST_PARAMS, DRIVE_ORDER_BY_VALUES } from '../utils.js';
import { downloadTextContent } from './text-content.js';
import { toNodeReadable } from '../utils/streams.js';
import { downloadDriveFile, GOOGLE_WORKSPACE_EXPORT_FORMATS } from '../download-file.js';
import { getSecureTokenPath } from '../auth/utils.js';
import { SCOPE_ALIASES, SCOPE_PRESETS, resolveOAuthScopes, splitScopes } from '../auth/scopes.js';
import { getActiveAuthMode, describeBypassedTokens, AUTH_MODE_OVERRIDE_ENV_VARS, type ActiveAuthMode } from '../auth/externalAuth.js';
import { getEffectiveIdentity } from '../auth/identity.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FOLDER_MIME_TYPE = 'application/vnd.google-apps.folder';
const SHORTCUT_MIME_TYPE = 'application/vnd.google-apps.shortcut';

// MIME types for binary file uploads (extension → MIME)
const BINARY_MIME_TYPES: Record<string, string> = {
  jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', gif: 'image/gif',
  webp: 'image/webp', svg: 'image/svg+xml', bmp: 'image/bmp', ico: 'image/x-icon',
  mp3: 'audio/mpeg', wav: 'audio/wav', ogg: 'audio/ogg', m4a: 'audio/mp4',
  aac: 'audio/aac', flac: 'audio/flac', opus: 'audio/opus',
  mp4: 'video/mp4', webm: 'video/webm', avi: 'video/x-msvideo', mov: 'video/quicktime',
  mkv: 'video/x-matroska', '3gp': 'video/3gpp',
  pdf: 'application/pdf', zip: 'application/zip', gz: 'application/gzip',
  tar: 'application/x-tar', json: 'application/json', xml: 'application/xml',
  csv: 'text/csv', html: 'text/html', css: 'text/css', js: 'application/javascript',
  doc: 'application/msword', docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xls: 'application/vnd.ms-excel', xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  ppt: 'application/vnd.ms-powerpoint', pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};

// ---------------------------------------------------------------------------
// Zod schemas
// ---------------------------------------------------------------------------

const SearchSchema = z.object({
  query: z.string().min(1, "Search query is required"),
  pageSize: z.number().int().min(1).max(100).optional(),
  pageToken: z.string().optional(),
  rawQuery: z.boolean().optional(),
  orderBy: z.enum(DRIVE_ORDER_BY_VALUES).optional().default("modifiedTime desc")
    .describe("Sort order for results."),
});

const CreateTextFileSchema = z.object({
  name: z.string().min(1, "File name is required"),
  content: z.string(),
  parentFolderId: z.string().optional()
});

const UpdateTextFileSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  content: z.string(),
  name: z.string().optional()
});

const ReadTextFileSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  maxLength: z.number().int().min(1).optional()
});

const CreateFolderSchema = z.object({
  name: z.string().min(1, "Folder name is required"),
  parent: z.string().optional()
});

const ListFolderSchema = z.object({
  folderId: z.string().optional(),
  pageSize: z.number().int().min(1).max(100).optional(),
  pageToken: z.string().optional()
});

const ListSharedDrivesSchema = z.object({
  pageSize: z.number().int().min(1).max(100).optional(),
  pageToken: z.string().optional()
});

const DeleteItemSchema = z.object({
  itemId: z.string().min(1, "Item ID is required")
});

const RenameItemSchema = z.object({
  itemId: z.string().min(1, "Item ID is required"),
  newName: z.string().min(1, "New name is required")
});

const MoveItemSchema = z.object({
  itemId: z.string().min(1, "Item ID is required"),
  destinationFolderId: z.string().optional()
});

const CopyFileSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  newName: z.string().optional(),
  parentFolderId: z.string().optional()
});

const CreateShortcutSchema = z.object({
  targetFileId: z.string().min(1, "Target file ID is required"),
  parentFolderId: z.string().optional(),
  shortcutName: z.string().optional()
});

const LockFileSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  reason: z.string().optional(),
  ownerRestricted: z.boolean().optional()
});

const UnlockFileSchema = z.object({
  fileId: z.string().min(1, "File ID is required")
});

const UploadFileSchema = z.object({
  localPath: z.string().min(1).optional(),
  contentBase64: z.string().min(1).optional(),
  fileId: z.string().min(1).optional(),
  name: z.string().optional(),
  parentFolderId: z.string().optional(),
  mimeType: z.string().optional(),
  convertToGoogleFormat: z.boolean().optional()
}).refine(
  (data) => !!data.localPath !== !!data.contentBase64,
  { message: "Provide exactly one of localPath or contentBase64" }
);

const DownloadFileSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  localPath: z.string().min(1, "Local file path is required"),
  exportMimeType: z.string().optional(),
  overwrite: z.boolean().optional().default(false),
});

const ListPermissionsSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
});

const AddPermissionSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  emailAddress: z.string().email("Valid email is required").optional(),
  domain: z.string().min(1, "Domain is required").optional(),
  role: z.enum(["owner", "organizer", "fileOrganizer", "writer", "commenter", "reader"]).default("reader"),
  type: z.enum(["user", "group", "domain", "anyone"]).default("user"),
  sendNotificationEmail: z.boolean().optional().default(false),
  emailMessage: z.string().optional(),
  allowFileDiscovery: z.boolean().optional(),
}).superRefine((data, ctx) => {
  if ((data.type === "user" || data.type === "group") && !data.emailAddress) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["emailAddress"],
      message: "emailAddress is required when type is 'user' or 'group'" });
  }
  if (data.type === "domain" && !data.domain) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["domain"],
      message: "domain is required when type is 'domain'" });
  }
});

const UpdatePermissionSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  permissionId: z.string().min(1, "Permission ID is required"),
  role: z.enum(["owner", "organizer", "fileOrganizer", "writer", "commenter", "reader"]),
});

const RemovePermissionSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  permissionId: z.string().optional(),
  emailAddress: z.string().email("Valid email is required").optional(),
}).superRefine((data, ctx) => {
  if (!data.permissionId && !data.emailAddress) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: "Either permissionId or emailAddress is required" });
  }
});

const ShareFileSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  emailAddress: z.string().email("Valid email is required"),
  role: z.enum(["writer", "commenter", "reader"]).default("reader"),
  sendNotificationEmail: z.boolean().optional().default(true),
  emailMessage: z.string().optional(),
});

const ConvertPdfToGoogleDocSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  newName: z.string().optional(),
  parentFolderId: z.string().optional(),
});

const BulkConvertFolderPdfsSchema = z.object({
  folderId: z.string().min(1, "Folder ID is required"),
  maxResults: z.number().int().min(1).max(200).optional().default(100),
  continueOnError: z.boolean().optional().default(true),
});

const UploadPdfWithSplitSchema = z.object({
  localPath: z.string().min(1, "Local file path is required"),
  split: z.boolean().optional().default(false),
  maxPagesPerChunk: z.number().int().min(1).max(500).optional(),
  parentFolderId: z.string().optional(),
  namePrefix: z.string().optional(),
});

async function splitPdfIntoChunkFiles(localPath: string, maxPagesPerChunk: number): Promise<{ tempDir: string; files: string[] }> {
  const sourceBytes = await readFile(localPath);
  const source = await PDFDocument.load(sourceBytes);
  const pageCount = source.getPageCount();

  if (pageCount === 0) {
    throw new Error('PDF contains no pages.');
  }

  const tempDir = await mkdtemp(join(tmpdir(), 'gdrive-mcp-split-'));
  const files: string[] = [];

  for (let start = 0, part = 1; start < pageCount; start += maxPagesPerChunk, part++) {
    const end = Math.min(start + maxPagesPerChunk, pageCount);
    const chunkDoc = await PDFDocument.create();
    const pages = await chunkDoc.copyPages(source, Array.from({ length: end - start }, (_, i) => start + i));
    for (const page of pages) chunkDoc.addPage(page);

    const chunkBytes = await chunkDoc.save();
    const chunkPath = join(tempDir, `part-${part}.pdf`);
    await writeFile(chunkPath, chunkBytes);
    files.push(chunkPath);
  }

  return { tempDir, files };
}

const GetRevisionsSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  pageSize: z.number().int().min(1).max(200).optional().default(50),
  pageToken: z.string().optional(),
});

const RestoreRevisionSchema = z.object({
  fileId: z.string().min(1, "File ID is required"),
  revisionId: z.string().min(1, "Revision ID is required"),
  confirm: z.boolean().optional().default(false),
});

const AuthTestFileAccessSchema = z.object({
  fileId: z.string().optional(),
});

const ManageAccountsSchema = z.object({
  action: z.enum(['list', 'add', 'remove', 'set_default']),
  // Nullable so a client can pass JSON `null` to `set_default` to clear the current
  // default (per the tool description); add/remove still reject a missing/null alias.
  account_id: z.string().nullable().optional(),
});

// Give the user up to 5 minutes to complete OAuth consent before giving up.
const ADD_ACCOUNT_CONSENT_TIMEOUT_MS = 5 * 60 * 1000;

/**
 * Race an in-flight account-add's `completion` against a timeout. On timeout (or any
 * failure) call `cancel()` to stop the embedded OAuth callback server, so we don't leak
 * the listening port or let a late consent silently register the account after we've
 * reported failure. `cancel()` is idempotent, so the success path's own teardown
 * (in `addAccountFlow`) is unaffected.
 */
export async function awaitConsentCompletion<T>(
  result: { completion: Promise<T>; cancel: () => Promise<void> },
  timeoutMs: number,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(
      () => reject(new Error(`Timed out after ${timeoutMs / 1000}s waiting for OAuth consent.`)),
      timeoutMs,
    );
  });
  try {
    return await Promise.race([result.completion, timeout]);
  } catch (err) {
    await result.cancel().catch(() => undefined);
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Map a `set_default` `account_id` to its target. A non-empty string is a real alias
 * (including the literal "null"); JSON `null`, `undefined`, or an empty string all clear
 * the current default.
 */
export function resolveSetDefaultTarget(account_id: string | null | undefined): string | null {
  return typeof account_id === 'string' && account_id.length > 0 ? account_id : null;
}

function getGrantedScopesFromAuthClient(ctx: ToolContext): string[] {
  const scopeRaw = ctx.authClient?.credentials?.scope;
  if (!scopeRaw || typeof scopeRaw !== 'string') return [];
  return splitScopes(scopeRaw);
}

function resolveScopeStatus(ctx: ToolContext): { requestedScopes: string[]; grantedScopes: string[]; missingScopes: string[] } {
  const requestedScopes = resolveOAuthScopes();
  const grantedScopes = getGrantedScopesFromAuthClient(ctx);
  const missingScopes = requestedScopes.filter((s) => !grantedScopes.includes(s));
  return { requestedScopes, grantedScopes, missingScopes };
}

export type AuthDiagnosticStatus =
  | 'needs_reauth'
  | 'scope_mismatch'
  | 'identity_error'
  | 'warning'
  | 'ok';

/**
 * The authGetStatus verdict, ordered most-actionable first. Extracted as a pure
 * function so the ladder's ordering — the subtle part — stays unit-testable.
 *
 * needs_reauth outranks everything in oauth mode: a missing token/refresh token
 * is the root cause and a failing about.get is just its symptom (a refresh
 * token is meaningless for service-account/external modes, so it's oauth-only).
 *
 * A *reliable* scope shortfall (grantedScopes known and short) outranks
 * identity_error: an under-scoped token's about.get 403 IS the symptom of the
 * missing scope, so re-auth-with-scopes is the real fix — don't send the user
 * chasing an identity problem. The reliability guard matters because
 * grantedScopes is often empty for disk-loaded tokens (making missingScopes
 * falsely list everything); when it's empty we fall through to identity_error
 * so a genuine wrong/empty identity isn't masked, and the trailing
 * scope_mismatch preserves prior behavior for that empty case. warning is last
 * so it never masks a specific diagnosis.
 */
export function computeAuthStatus(input: {
  authMode: ActiveAuthMode;
  tokenFileExists: boolean;
  hasRefreshToken: boolean;
  grantedScopes: string[];
  missingScopes: string[];
  identityError: boolean;
  warningCount: number;
}): AuthDiagnosticStatus {
  const scopesReliable = input.grantedScopes.length > 0;
  return (input.authMode === 'oauth' && (!input.tokenFileExists || !input.hasRefreshToken)) ? 'needs_reauth'
    : (scopesReliable && input.missingScopes.length > 0) ? 'scope_mismatch'
    : input.identityError ? 'identity_error'
    : input.missingScopes.length > 0 ? 'scope_mismatch'
    : input.warningCount > 0 ? 'warning'
    : 'ok';
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

export const toolDefinitions: ToolDefinition[] = [
  {
    name: "search",
    description: "Search for files in Google Drive. Results are sorted most-recently-modified first by default; use orderBy for a different order. Set rawQuery=true to pass a raw Google Drive API query supporting operators like modifiedTime, createdTime, mimeType, name contains, etc.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query. When rawQuery=true, this is passed directly to the Google Drive API as the q parameter." },
        pageSize: { type: "number", description: "Results per page (default 50, max 100)" },
        pageToken: { type: "string", description: "Token for next page of results" },
        rawQuery: { type: "boolean", description: "If true, pass query directly to Google Drive API without wrapping in fullText contains. Enables date filters, mimeType filters, etc." },
        orderBy: { type: "string", description: "Sort order for results. Keys without 'desc' sort ascending, so 'modifiedTime' is oldest-first. When paging, repeat the same value on every pageToken call.", enum: [...DRIVE_ORDER_BY_VALUES], default: "modifiedTime desc" },
      },
      required: ["query"],
    },
  },
  {
    name: "createTextFile",
    description: "Create a new text or markdown file",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "File name (.txt or .md)" },
        content: { type: "string", description: "File content" },
        parentFolderId: { type: "string", description: "Parent folder ID" }
      },
      required: ["name", "content"]
    }
  },
  {
    name: "updateTextFile",
    description: "Update an existing text or markdown file",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "ID of the file to update" },
        content: { type: "string", description: "New file content" },
        name: { type: "string", description: "New name (.txt or .md)" }
      },
      required: ["fileId", "content"]
    }
  },
  {
    name: "readTextFile",
    description: "Read content of a text file (any text/* MIME type, e.g. text/plain, text/markdown, text/csv). For Google Docs, use readGoogleDoc.",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "ID of the file to read" },
        maxLength: { type: "number", description: "Maximum number of characters (Unicode code points) to return; content beyond this is truncated" }
      },
      required: ["fileId"]
    }
  },
  {
    name: "createFolder",
    description: "Create a new folder in Google Drive",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Folder name" },
        parent: { type: "string", description: "Parent folder ID or path" }
      },
      required: ["name"]
    }
  },
  {
    name: "listFolder",
    description: "List contents of a folder (defaults to root)",
    inputSchema: {
      type: "object",
      properties: {
        folderId: { type: "string", description: "Folder ID" },
        pageSize: { type: "number", description: "Items to return (default 50, max 100)" },
        pageToken: { type: "string", description: "Token for next page" }
      }
    }
  },
  {
    name: "listSharedDrives",
    description: "List available Google Shared Drives",
    inputSchema: {
      type: "object",
      properties: {
        pageSize: { type: "number", description: "Drives to return (default 50, max 100)" },
        pageToken: { type: "string", description: "Token for next page" }
      }
    }
  },
  {
    name: "deleteItem",
    description: "Move a file or folder to trash (can be restored from Google Drive trash)",
    inputSchema: {
      type: "object",
      properties: {
        itemId: { type: "string", description: "ID of the item to delete" }
      },
      required: ["itemId"]
    }
  },
  {
    name: "renameItem",
    description: "Rename a file or folder",
    inputSchema: {
      type: "object",
      properties: {
        itemId: { type: "string", description: "ID of the item to rename" },
        newName: { type: "string", description: "New name" }
      },
      required: ["itemId", "newName"]
    }
  },
  {
    name: "moveItem",
    description: "Move a file or folder",
    inputSchema: {
      type: "object",
      properties: {
        itemId: { type: "string", description: "ID of the item to move" },
        destinationFolderId: { type: "string", description: "Destination folder ID" }
      },
      required: ["itemId"]
    }
  },
  {
    name: "copyFile",
    description: "Creates a copy of a Google Drive file or document",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "ID of the file to copy" },
        newName: { type: "string", description: "Name for the copied file. If not provided, will use 'Copy of [original name]'" },
        parentFolderId: { type: "string", description: "ID or path of the destination folder (defaults to same folder as original)" }
      },
      required: ["fileId"]
    }
  },
  {
    name: "uploadFile",
    description: "Upload a file (any type: image, audio, video, PDF, etc.) to Google Drive, either from a local path on the server or from base64-encoded content. When fileId is provided, uploads the content as a new version of that existing file (in-place update) instead of creating a new file.",
    inputSchema: {
      type: "object",
      properties: {
        localPath: { type: "string", description: "Absolute path to the local file to upload (on the machine running this server). Provide either localPath or contentBase64." },
        contentBase64: { type: "string", description: "Base64-encoded file content. Alternative to localPath for clients without access to the server's filesystem. Provide either localPath or contentBase64." },
        fileId: { type: "string", description: "ID of an existing Drive file to update in place: the uploaded content becomes a new version of this file (same ID, revision history preserved). Omit to create a new file." },
        name: { type: "string", description: "File name in Drive (defaults to local filename; required when creating a new file from contentBase64)" },
        parentFolderId: { type: "string", description: "Parent folder ID or path (e.g., '/Work/Projects'). Creates folders if needed. Defaults to root. Not allowed with fileId (use moveItem to move a file)." },
        mimeType: { type: "string", description: "MIME type (auto-detected from extension if omitted)" },
        convertToGoogleFormat: { type: "boolean", description: "Convert uploaded file to Google Workspace format (e.g., .docx to Google Doc, .xlsx to Google Sheet, .pptx to Google Slides). Defaults to false. Not allowed with fileId." }
      },
      required: []
    }
  },
  {
    name: "downloadFile",
    description: "Download a Google Drive file to a local path. For Google Workspace files (Docs, Sheets, Slides, Drawings), exports to the specified format. For regular files, downloads as-is. Streams directly to disk.",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Google Drive file ID" },
        localPath: { type: "string", description: "Absolute local path to save the file (must start with /). Can be a directory (filename auto-resolved from Drive metadata) or a full file path. Path is normalized before use." },
        exportMimeType: {
          type: "string",
          description: "For Google Workspace files: MIME type to export as (e.g., 'application/pdf', 'text/csv'). Auto-detected from file extension if omitted. Ignored for non-Workspace files."
        },
        overwrite: {
          type: "boolean",
          description: "Whether to overwrite if file already exists at localPath. When false (default), returns an error instead of replacing the file."
        }
      },
      required: ["fileId", "localPath"]
    }
  },
  {
    name: "listPermissions",
    description: "List sharing permissions for a file",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Google Drive file ID" }
      },
      required: ["fileId"]
    }
  },
  {
    name: "addPermission",
    description: "Add a sharing permission to a file. If the principal already holds a permission on the file, Drive updates that permission instead of creating a second one, and the requested role is applied even when it is a downgrade (asking for 'reader' for an existing 'writer' leaves them as 'reader'). Provide emailAddress for type 'user'/'group', domain for type 'domain', and neither for 'anyone'.",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Google Drive file ID" },
        emailAddress: { type: "string", description: "Target user/group email. Required when type is 'user' or 'group'; ignored for 'domain'/'anyone'." },
        domain: { type: "string", description: "Target domain, e.g. 'example.com'. Required when type is 'domain'; ignored otherwise." },
        role: { type: "string", enum: ["owner", "organizer", "fileOrganizer", "writer", "commenter", "reader"], description: "Permission role" },
        type: { type: "string", enum: ["user", "group", "domain", "anyone"], description: "Principal type" },
        sendNotificationEmail: { type: "boolean", description: "Send notification email" },
        emailMessage: { type: "string", description: "Custom message to include in the notification email. Ignored unless sendNotificationEmail is true." },
        allowFileDiscovery: { type: "boolean", description: "Only for type 'domain'/'anyone': false (default) = accessible with the link only; true = discoverable in search. Ignored for 'user'/'group'." }
      },
      required: ["fileId"]
    }
  },
  {
    name: "updatePermission",
    description: "Update an existing permission role",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Google Drive file ID" },
        permissionId: { type: "string", description: "Permission ID" },
        role: { type: "string", enum: ["owner", "organizer", "fileOrganizer", "writer", "commenter", "reader"], description: "New role" }
      },
      required: ["fileId", "permissionId", "role"]
    }
  },
  {
    name: "removePermission",
    description: "Remove a permission from a file (by permissionId or emailAddress)",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Google Drive file ID" },
        permissionId: { type: "string", description: "Permission ID" },
        emailAddress: { type: "string", description: "User email (alternative to permissionId)" }
      },
      required: ["fileId"]
    }
  },
  {
    name: "shareFile",
    description: "Convenience wrapper to share a file with a user email. If the user already holds a permission on the file, that permission's role is set to the requested one (even when it is a downgrade) instead of creating a duplicate.",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Google Drive file ID" },
        emailAddress: { type: "string", description: "User email" },
        role: { type: "string", enum: ["writer", "commenter", "reader"], description: "Access role" },
        sendNotificationEmail: { type: "boolean", description: "Send notification email" },
        emailMessage: { type: "string", description: "Custom message to include in the notification email. Ignored unless sendNotificationEmail is true." }
      },
      required: ["fileId", "emailAddress"]
    }
  },
  {
    name: "convertPdfToGoogleDoc",
    description: "Convert an existing PDF in Drive into an editable Google Doc",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "PDF file ID in Google Drive" },
        newName: { type: "string", description: "Optional name for converted Doc" },
        parentFolderId: { type: "string", description: "Optional destination folder ID" }
      },
      required: ["fileId"]
    }
  },
  {
    name: "bulkConvertFolderPdfs",
    description: "Convert all PDFs in a folder into Google Docs and return per-file results",
    inputSchema: {
      type: "object",
      properties: {
        folderId: { type: "string", description: "Folder ID containing PDFs" },
        maxResults: { type: "number", description: "Maximum PDFs to process (1-200, default: 100)" },
        continueOnError: { type: "boolean", description: "Continue conversion when one file fails (default: true)" }
      },
      required: ["folderId"]
    }
  },
  {
    name: "uploadPdfWithSplit",
    description: "Upload PDF and optionally split into chunked parts (metadata split plan for now)",
    inputSchema: {
      type: "object",
      properties: {
        localPath: { type: "string", description: "Absolute path to local PDF" },
        split: { type: "boolean", description: "Enable split mode" },
        maxPagesPerChunk: { type: "number", description: "Target max pages per chunk (advisory metadata)" },
        parentFolderId: { type: "string", description: "Optional destination folder ID" },
        namePrefix: { type: "string", description: "Optional output name prefix" }
      },
      required: ["localPath"]
    }
  },
  {
    name: "getRevisions",
    description: "List revisions for a file",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Google Drive file ID" },
        pageSize: { type: "number", description: "Max revisions to return (default 50, max 200)" },
        pageToken: { type: "string", description: "Page token for pagination" }
      },
      required: ["fileId"]
    }
  },
  {
    name: "restoreRevision",
    description: "Restore a file to a selected revision (creates a new head revision). Note: workspace files (Docs, Sheets, Slides) are restored via export/import and may lose some formatting.",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Google Drive file ID" },
        revisionId: { type: "string", description: "Revision ID to restore" },
        confirm: { type: "boolean", description: "Safety flag. Must be true to execute restore." }
      },
      required: ["fileId", "revisionId"]
    }
  },
  {
    name: "manage_accounts",
    description:
      "Manage connected Google accounts. action='list' shows all connected accounts (no secrets). action='add' starts an OAuth flow for a new account identified by account_id (alias). action='remove' deletes local tokens for an account. action='set_default' picks which account is used when tool calls omit the `account` parameter.",
    inputSchema: {
      type: "object",
      properties: {
        action: {
          type: "string",
          enum: ["list", "add", "remove", "set_default"],
          description: "Which account-lifecycle operation to perform."
        },
        account_id: {
          type: ["string", "null"],
          description:
            "Account alias. Required for add/remove/set_default. Must match /^[a-z0-9][a-z0-9_-]{0,31}$/ and not be reserved (e.g. 'default', 'all', '*'). Pass JSON null with set_default to clear the current default."
        }
      },
      required: ["action"]
    }
  },
  {
    name: "authGetStatus",
    description: "Show authentication/token status and scope diagnostics",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "authListScopes",
    description: "List configured/requested scopes and currently granted scopes",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "authTestFileAccess",
    description: "Run auth diagnostics against Drive API/file access",
    inputSchema: {
      type: "object",
      properties: {
        fileId: { type: "string", description: "Optional file ID for targeted access check" }
      }
    }
  },
  {
    name: "createShortcut",
    description: "Create a shortcut (link) to a file or folder in Google Drive. Useful for referencing the same document from multiple locations without duplicating it.",
    inputSchema: {
      type: "object",
      properties: {
        targetFileId: {
          type: "string",
          description: "The file or folder ID (not a path) to create a shortcut to"
        },
        parentFolderId: {
          type: "string",
          description: "ID or path of the folder where the shortcut will be created"
        },
        shortcutName: {
          type: "string",
          description: "Custom name for the shortcut (defaults to original file name)"
        }
      },
      required: ["targetFileId"]
    }
  },
  {
    name: "lockFile",
    description: "Lock a file to prevent editing by setting content restrictions. The file remains readable but cannot be modified until unlocked.",
    inputSchema: {
      type: "object",
      properties: {
        fileId: {
          type: "string",
          description: "ID of the file to lock"
        },
        reason: {
          type: "string",
          description: "Reason for locking the file (shown to users who try to edit)"
        },
        ownerRestricted: {
          type: "boolean",
          description: "If true, only the file owner can unlock the file (default: false)"
        }
      },
      required: ["fileId"]
    }
  },
  {
    name: "unlockFile",
    description: "Unlock a previously locked file by removing content restrictions, restoring full edit access.",
    inputSchema: {
      type: "object",
      properties: {
        fileId: {
          type: "string",
          description: "ID of the file to unlock"
        }
      },
      required: ["fileId"]
    }
  },
];

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Permission writes
// ---------------------------------------------------------------------------
// Drive can apply a different role than the one requested: `permissions.create`
// upserts onto a permission the principal already holds, and the resulting role
// is not always the requested one. Every permission write below compares the
// role Drive returned against the request and never reports the request as
// the result.

function describeRole(role: string | null | undefined): string {
  return role ? `'${role}'` : 'no role';
}

type PermissionRoleOutcome =
  | { ok: true; role: string; corrected: false }
  | { ok: true; role: string; corrected: true; initiallyApplied: string | null }
  | { ok: false; detail: string };

/**
 * Reconcile the role Drive applied on a `permissions.create` with the role
 * requested. On a mismatch, issue one corrective `permissions.update` and
 * report what happened: success with the role Drive first applied (so the
 * caller learns an existing permission was changed, including a downgrade),
 * or a failure describing what Drive holds now.
 */
async function reconcileCreatedPermissionRole(
  drive: drive_v3.Drive,
  fileId: string,
  created: drive_v3.Schema$Permission,
  requestedRole: string,
): Promise<PermissionRoleOutcome> {
  if (created.role === requestedRole) return { ok: true, role: requestedRole, corrected: false };
  const applied = describeRole(created.role);
  if (!created.id) {
    return { ok: false, detail: `requested '${requestedRole}' but Drive applied ${applied} and returned no permission id to correct` };
  }
  try {
    const corrected = await drive.permissions.update({
      fileId,
      permissionId: created.id,
      requestBody: { role: requestedRole },
      fields: 'id,role',
      supportsAllDrives: true,
    });
    if (corrected.data.role === requestedRole) {
      return { ok: true, role: requestedRole, corrected: true, initiallyApplied: created.role ?? null };
    }
    return { ok: false, detail: `requested '${requestedRole}' but Drive holds ${describeRole(corrected.data.role)} even after a corrective update` };
  } catch (e: unknown) {
    return { ok: false, detail: `requested '${requestedRole}' but Drive applied ${applied}, and the corrective update failed: ${e instanceof Error ? e.message : String(e)}` };
  }
}

export async function handleTool(
  toolName: string,
  args: Record<string, unknown>,
  ctx: ToolContext,
): Promise<ToolResult | null> {
  switch (toolName) {

    case "search": {
      const validation = SearchSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const { query: userQuery, pageSize, pageToken, rawQuery, orderBy } = validation.data;

      let formattedQuery: string;
      if (rawQuery) {
        // Use query directly; auto-append trashed guard unless user already includes it
        formattedQuery = /\btrashed\s*=/.test(userQuery)
          ? userQuery
          : `${userQuery} and trashed = false`;
      } else {
        const escapedQuery = escapeDriveQuery(userQuery);
        formattedQuery = `fullText contains '${escapedQuery}' and trashed = false`;
      }

      const res = await ctx.getDrive().files.list({
        q: formattedQuery,
        pageSize: Math.min(pageSize || 50, 100),
        pageToken: pageToken,
        fields: "nextPageToken, files(id, name, mimeType, createdTime, modifiedTime, size, parents)",
        orderBy,
        ...ALL_DRIVES_LIST_PARAMS
      });

      // Resolve folder paths from parent IDs (with dedup for concurrent lookups)
      const pathCache: Record<string, Promise<string>> = {};
      function resolveParentPath(folderId: string, depth = 0): Promise<string> {
        if (depth >= 10) return Promise.resolve(folderId);
        if (folderId in pathCache) return pathCache[folderId];
        const promise = (async () => {
          try {
            const folderRes = await ctx.getDrive().files.get({
              fileId: folderId,
              fields: "name, parents",
              supportsAllDrives: true,
            });
            const name = folderRes.data.name || folderId;
            const parents = folderRes.data.parents;
            if (parents && parents.length > 0 && parents[0] !== folderId) {
              const parentPath = await resolveParentPath(parents[0], depth + 1);
              return `${parentPath}/${name}`;
            }
            return name;
          } catch {
            return folderId;
          }
        })();
        pathCache[folderId] = promise;
        return promise;
      }

      const files = res.data.files || [];
      const fileLines = await Promise.all(
        files.map(async (f: drive_v3.Schema$File) => {
          let folderPath = '';
          if (f.parents && f.parents.length > 0) {
            folderPath = await resolveParentPath(f.parents[0]);
          }
          return `${f.name} (${f.mimeType}) [id: ${f.id}, path: ${folderPath || '/'}] [created: ${f.createdTime || 'N/A'}, modified: ${f.modifiedTime || 'N/A'}]`;
        }),
      );

      ctx.log('Search results', { query: userQuery, rawQuery: !!rawQuery, orderBy, resultCount: files.length });

      // Name the ordering in the response: an unlabeled list invites the caller to
      // pick "the most recent" by eye and get it wrong (#167).
      let response = `Found ${files.length} files (ordered by ${orderBy}):\n${fileLines.join("\n")}`;
      if (res.data.nextPageToken) {
        // Keep the token alone on its line — trailing text after an unquoted token
        // invites the caller to carry it into the token value.
        response += `\n\nMore results available. Use pageToken: ${res.data.nextPageToken}`;
        // orderBy has a default, so a follow-up call carrying only the token would
        // silently fall back to `modifiedTime desc` and re-sort page 2 (#167).
        response += `\nPass orderBy: ${orderBy} again to keep this ordering.`;
      }

      return {
        content: [{ type: "text", text: response }],
        isError: false,
      };
    }

    case "createTextFile": {
      const validation = CreateTextFileSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      ctx.validateTextFileExtension(data.name);
      const parentFolderId = await ctx.resolveFolderId(data.parentFolderId);

      // Check if file already exists
      const existingFileId = await ctx.checkFileExists(data.name, parentFolderId);
      if (existingFileId) {
        return errorResponse(
          `A file named "${data.name}" already exists in this location. ` +
          `To update it, use updateTextFile with fileId: ${existingFileId}`
        );
      }

      const fileMetadata = {
        name: data.name,
        mimeType: getMimeTypeFromFilename(data.name),
        parents: [parentFolderId]
      };

      const file = await ctx.getDrive().files.create({
        requestBody: fileMetadata,
        media: {
          mimeType: fileMetadata.mimeType,
          body: data.content,
        },
        supportsAllDrives: true
      });

      ctx.log('File created successfully', { fileId: file.data?.id });
      return {
        content: [{
          type: "text",
          text: `Created file: ${file.data?.name || data.name}\nID: ${file.data?.id || 'unknown'}`
        }],
        isError: false
      };
    }

    case "updateTextFile": {
      const validation = UpdateTextFileSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      // Check file MIME type
      const existingFile = await ctx.getDrive().files.get({
        fileId: data.fileId,
        fields: 'mimeType, name, parents',
        supportsAllDrives: true
      });

      const currentMimeType = existingFile.data.mimeType || 'text/plain';
      if (!isTextMime(currentMimeType)) {
        return errorResponse("File is not a text file (expected a text/* MIME type).");
      }

      const updateMetadata: { name?: string; mimeType?: string } = {};
      if (data.name) {
        ctx.validateTextFileExtension(data.name);
        updateMetadata.name = data.name;
        updateMetadata.mimeType = getMimeTypeFromFilename(data.name);
      }

      const updatedFile = await ctx.getDrive().files.update({
        fileId: data.fileId,
        requestBody: updateMetadata,
        media: {
          mimeType: updateMetadata.mimeType || currentMimeType,
          body: data.content
        },
        fields: 'id, name, modifiedTime, webViewLink',
        supportsAllDrives: true
      });

      return {
        content: [{
          type: "text",
          text: `Updated file: ${updatedFile.data.name}\nModified: ${updatedFile.data.modifiedTime}`
        }],
        isError: false
      };
    }

    case "readTextFile": {
      const validation = ReadTextFileSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      const metadata = await ctx.getDrive().files.get({
        fileId: data.fileId,
        fields: 'mimeType, name',
        supportsAllDrives: true
      });

      const mimeType = metadata.data.mimeType || '';
      const fileName = metadata.data.name || 'unknown';

      if (!isTextMime(mimeType)) {
        return errorResponse(
          `File "${fileName}" has MIME type "${mimeType}", which is not a text/* type. ` +
          `For Google Docs, use readGoogleDoc instead.`
        );
      }

      const fullContent = await downloadTextContent(ctx.getDrive(), data.fileId);
      // Length/truncation are measured in Unicode code points (whole characters),
      // so truncation never splits a surrogate pair and the reported length matches
      // the 0-based code-point offsets used by insertText/deleteRange.
      const codePoints = Array.from(fullContent);
      const originalLength = codePoints.length;
      const truncated = data.maxLength !== undefined && originalLength > data.maxLength;
      const content = truncated ? codePoints.slice(0, data.maxLength).join('') : fullContent;

      const header =
        `File: ${fileName}\n` +
        `MIME type: ${mimeType}\n` +
        `Length: ${originalLength} characters\n` +
        `Truncated: ${truncated ? `yes (showing first ${data.maxLength})` : 'no'}\n` +
        `---\n`;

      return {
        content: [{ type: "text", text: header + content }],
        isError: false
      };
    }

    case "createFolder": {
      const validation = CreateFolderSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      const parentFolderId = await ctx.resolveFolderId(data.parent);

      // Check if folder already exists
      const existingFolderId = await ctx.checkFileExists(data.name, parentFolderId);
      if (existingFolderId) {
        return errorResponse(
          `A folder named "${data.name}" already exists in this location. ` +
          `Folder ID: ${existingFolderId}`
        );
      }
      const folderMetadata = {
        name: data.name,
        mimeType: FOLDER_MIME_TYPE,
        parents: [parentFolderId]
      };

      const folder = await ctx.getDrive().files.create({
        requestBody: folderMetadata,
        fields: 'id, name, webViewLink',
        supportsAllDrives: true
      });

      ctx.log('Folder created successfully', { folderId: folder.data.id, name: folder.data.name });

      return {
        content: [{
          type: "text",
          text: `Created folder: ${folder.data.name}\nID: ${folder.data.id}`
        }],
        isError: false
      };
    }

    case "listFolder": {
      const validation = ListFolderSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      // Default to root if no folder specified
      const targetFolderId = data.folderId || 'root';

      const res = await ctx.getDrive().files.list({
        q: `'${targetFolderId}' in parents and trashed = false`,
        pageSize: Math.min(data.pageSize || 50, 100),
        pageToken: data.pageToken,
        fields: "nextPageToken, files(id, name, mimeType, modifiedTime, size)",
        orderBy: "name",
        // Parent-scoped: two flags only, no corpora=allDrives, so the listing
        // can never come back as an incompleteSearch partial result (#137).
        ...PARENT_SCOPED_LIST_PARAMS
      });

      const files = res.data.files || [];
      const formattedFiles = files.map((file: drive_v3.Schema$File) => {
        const isFolder = file.mimeType === FOLDER_MIME_TYPE;
        return `${isFolder ? '📁' : '📄'} ${file.name} (ID: ${file.id})`;
      }).join('\n');

      let response = `Contents of folder:\n\n${formattedFiles}`;
      if (res.data.nextPageToken) {
        response += `\n\nMore items available. Use pageToken: ${res.data.nextPageToken}`;
      }

      return {
        content: [{ type: "text", text: response }],
        isError: false
      };
    }

    case "listSharedDrives": {
      const validation = ListSharedDrivesSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      const res = await ctx.getDrive().drives.list({
        pageSize: Math.min(data.pageSize || 50, 100),
        pageToken: data.pageToken,
        fields: 'nextPageToken, drives(id, name, createdTime, hidden)'
      });

      const drives = res.data.drives || [];
      if (drives.length === 0) {
        return { content: [{ type: 'text', text: 'No shared drives found.' }], isError: false };
      }

      const formatted = drives
        .map((d: drive_v3.Schema$Drive) => `${d.name} (ID: ${d.id}${d.hidden ? ', hidden' : ''})`)
        .join('\n');

      let response = `Found ${drives.length} shared drives:\n${formatted}`;
      if (res.data.nextPageToken) {
        response += `\n\nMore results available. Use pageToken: ${res.data.nextPageToken}`;
      }

      return {
        content: [{ type: 'text', text: response }],
        isError: false,
      };
    }

    case "deleteItem": {
      const validation = DeleteItemSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      const item = await ctx.getDrive().files.get({ fileId: data.itemId, fields: 'name', supportsAllDrives: true });

      // Move to trash instead of permanent deletion
      await ctx.getDrive().files.update({
        fileId: data.itemId,
        requestBody: {
          trashed: true
        },
        supportsAllDrives: true
      });

      ctx.log('Item moved to trash successfully', { itemId: data.itemId, name: item.data.name });
      return {
        content: [{ type: "text", text: `Successfully moved to trash: ${item.data.name}` }],
        isError: false
      };
    }

    case "renameItem": {
      const validation = RenameItemSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      // If it's a text file, check extension
      const item = await ctx.getDrive().files.get({ fileId: data.itemId, fields: 'name, mimeType', supportsAllDrives: true });
      if (Object.values(TEXT_MIME_TYPES).includes(item.data.mimeType || '')) {
        ctx.validateTextFileExtension(data.newName);
      }

      const updatedItem = await ctx.getDrive().files.update({
        fileId: data.itemId,
        requestBody: { name: data.newName },
        fields: 'id, name, modifiedTime',
        supportsAllDrives: true
      });

      return {
        content: [{
          type: "text",
          text: `Successfully renamed "${item.data.name}" to "${updatedItem.data.name}"`
        }],
        isError: false
      };
    }

    case "moveItem": {
      const validation = MoveItemSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      const destinationFolderId = data.destinationFolderId ?
        await ctx.resolveFolderId(data.destinationFolderId) :
        'root';

      // Check we aren't moving a folder into itself or its descendant
      if (data.destinationFolderId === data.itemId) {
        return errorResponse("Cannot move a folder into itself.");
      }

      const item = await ctx.getDrive().files.get({ fileId: data.itemId, fields: 'name, parents', supportsAllDrives: true });

      // Perform move
      await ctx.getDrive().files.update({
        fileId: data.itemId,
        addParents: destinationFolderId,
        removeParents: item.data.parents?.join(',') || '',
        fields: 'id, name, parents',
        supportsAllDrives: true
      });

      // Get the destination folder name for a nice response
      const destinationFolder = await ctx.getDrive().files.get({
        fileId: destinationFolderId,
        fields: 'name',
        supportsAllDrives: true
      });

      return {
        content: [{
          type: "text",
          text: `Successfully moved "${item.data.name}" to "${destinationFolder.data.name}"`
        }],
        isError: false
      };
    }

    case "copyFile": {
      const validation = CopyFileSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      // Get original file info
      const originalFile = await ctx.getDrive().files.get({
        fileId: data.fileId,
        fields: 'name,parents',
        supportsAllDrives: true
      });

      const copyMetadata: any = {
        name: data.newName || `Copy of ${originalFile.data.name}`
      };

      if (data.parentFolderId) {
        const resolvedParentId = await ctx.resolveFolderId(data.parentFolderId);
        copyMetadata.parents = [resolvedParentId];
      } else if (originalFile.data.parents) {
        copyMetadata.parents = originalFile.data.parents;
      }

      const response = await ctx.getDrive().files.copy({
        fileId: data.fileId,
        requestBody: copyMetadata,
        fields: 'id,name,webViewLink,parents',
        supportsAllDrives: true
      });

      return {
        content: [{ type: "text", text: `Successfully copied file as "${response.data.name}"\nNew file ID: ${response.data.id}\nLink: ${response.data.webViewLink}` }],
        isError: false
      };
    }

    case "createShortcut": {
      const validation = CreateShortcutSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      const parentId = await ctx.resolveFolderId(data.parentFolderId);

      // Get target file metadata for default name
      const targetFile = await ctx.getDrive().files.get({
        fileId: data.targetFileId,
        fields: 'id, name, mimeType',
        supportsAllDrives: true
      });

      const shortcutName = data.shortcutName || targetFile.data.name || 'Shortcut';

      const shortcut = await ctx.getDrive().files.create({
        requestBody: {
          name: shortcutName,
          mimeType: SHORTCUT_MIME_TYPE,
          shortcutDetails: {
            targetId: data.targetFileId
          },
          parents: [parentId]
        },
        fields: 'id, name, webViewLink, shortcutDetails',
        supportsAllDrives: true
      });

      ctx.log('Shortcut created', {
        shortcutId: shortcut.data.id,
        targetId: data.targetFileId,
        name: shortcutName
      });

      return {
        content: [{
          type: "text",
          text: `Shortcut created successfully!\n\nShortcut: ${shortcut.data.name} (${shortcut.data.id})\nTarget: ${targetFile.data.name} (${data.targetFileId})\nLocation: folder ${parentId}\nLink: ${shortcut.data.webViewLink || 'N/A'}`
        }],
        isError: false
      };
    }

    case "lockFile": {
      const validation = LockFileSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      const fileInfo = await ctx.getDrive().files.get({
        fileId: data.fileId,
        fields: 'id, name, contentRestrictions',
        supportsAllDrives: true
      });

      const existingRestrictions = fileInfo.data.contentRestrictions || [];
      if (existingRestrictions.some((r) => r.readOnly)) {
        return {
          content: [{
            type: "text",
            text: `File "${fileInfo.data.name}" is already locked.`
          }],
          isError: false
        };
      }

      await ctx.getDrive().files.update({
        fileId: data.fileId,
        requestBody: {
          contentRestrictions: [{
            readOnly: true,
            reason: data.reason || 'Locked via MCP',
            ownerRestricted: data.ownerRestricted ?? false
          }]
        },
        supportsAllDrives: true
      });

      ctx.log('File locked', { fileId: data.fileId, name: fileInfo.data.name, reason: data.reason });

      return {
        content: [{
          type: "text",
          text: `File locked successfully!\n\nFile: ${fileInfo.data.name}\nReason: ${data.reason || 'Locked via MCP'}${data.ownerRestricted ? '\nOwner-restricted: only the file owner can unlock' : ''}\n\nThe file is now read-only and cannot be edited or deleted.`
        }],
        isError: false
      };
    }

    case "unlockFile": {
      const validation = UnlockFileSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      const fileInfo = await ctx.getDrive().files.get({
        fileId: data.fileId,
        fields: 'id, name, contentRestrictions',
        supportsAllDrives: true
      });

      const existingRestrictions = fileInfo.data.contentRestrictions || [];
      if (!existingRestrictions.some((r) => r.readOnly)) {
        return {
          content: [{
            type: "text",
            text: `File "${fileInfo.data.name}" is not locked.`
          }],
          isError: false
        };
      }

      await ctx.getDrive().files.update({
        fileId: data.fileId,
        requestBody: {
          contentRestrictions: [{ readOnly: false }]
        },
        supportsAllDrives: true
      });

      ctx.log('File unlocked', { fileId: data.fileId, name: fileInfo.data.name });

      return {
        content: [{
          type: "text",
          text: `File unlocked successfully!\n\nFile: ${fileInfo.data.name}\n\nThe file can now be edited and deleted.`
        }],
        isError: false
      };
    }

    case "uploadFile": {
      const validation = UploadFileSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;

      if (data.fileId && data.convertToGoogleFormat) {
        return errorResponse('convertToGoogleFormat cannot be combined with fileId (in-place update).');
      }
      if (data.fileId && data.parentFolderId) {
        return errorResponse('parentFolderId cannot be combined with fileId (in-place update). Use moveItem to move a file.');
      }

      let contentBuffer: Buffer | undefined;
      let contentSize: number;
      if (data.localPath) {
        // Validate local file exists
        if (!existsSync(data.localPath)) {
          return errorResponse(`File not found: ${data.localPath}`);
        }
        contentSize = statSync(data.localPath).size;
      } else {
        // Node's base64 decoder silently drops invalid characters rather than
        // throwing, so validate the payload before decoding to avoid uploading
        // truncated/empty content as a "success".
        const b64 = data.contentBase64!.replace(/\s/g, '');
        if (!b64 || b64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(b64)) {
          return errorResponse('contentBase64 is not valid base64-encoded content.');
        }
        contentBuffer = Buffer.from(b64, 'base64');
        contentSize = contentBuffer.length;
      }
      const mediaBody = () => contentBuffer ? Readable.from(contentBuffer) : createReadStream(data.localPath!);

      const fileName = data.name || data.localPath?.split(/[\\/]/).pop() || '';
      if (!fileName && !data.fileId) {
        return errorResponse('name is required when creating a new file from contentBase64.');
      }
      const ext = fileName.split('.').pop()?.toLowerCase() || '';
      let detectedMime = data.mimeType || BINARY_MIME_TYPES[ext] || '';
      if (!detectedMime && data.fileId) {
        // In-place update with no MIME hint: reuse the existing file's MIME type
        const existing = await ctx.getDrive().files.get({
          fileId: data.fileId,
          fields: 'mimeType',
          supportsAllDrives: true
        });
        const existingMime = existing.data.mimeType || '';
        if (existingMime.startsWith('application/vnd.google-apps')) {
          return errorResponse(
            `File ${data.fileId} is a Google Workspace file (${existingMime}). ` +
            `Specify mimeType (e.g. the Office format of the uploaded content) so Drive can convert it.`
          );
        }
        detectedMime = existingMime;
      }
      if (!detectedMime) {
        detectedMime = 'application/octet-stream';
      }

      // Google Workspace conversion mapping
      const GOOGLE_FORMAT_MAP: Record<string, string> = {
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'application/vnd.google-apps.document',
        'application/msword': 'application/vnd.google-apps.document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'application/vnd.google-apps.spreadsheet',
        'application/vnd.ms-excel': 'application/vnd.google-apps.spreadsheet',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'application/vnd.google-apps.presentation',
        'application/vnd.ms-powerpoint': 'application/vnd.google-apps.presentation',
      };

      const targetMimeType = data.convertToGoogleFormat ? GOOGLE_FORMAT_MAP[detectedMime] : undefined;

      if (data.convertToGoogleFormat && !targetMimeType) {
        return errorResponse(
          `Cannot convert MIME type "${detectedMime}" to a Google Workspace format. ` +
          `Supported: .docx, .doc, .xlsx, .xls, .pptx, .ppt`
        );
      }

      const uploadName = targetMimeType ? fileName.replace(/\.[^.]+$/, '') : fileName;

      ctx.log('Uploading file', { localPath: data.localPath, fileId: data.fileId, name: uploadName, mimeType: detectedMime, convertToGoogle: !!targetMimeType, size: contentSize });

      let file: { data?: drive_v3.Schema$File };
      if (data.fileId) {
        // In-place update: upload content as a new version of the existing file.
        // The stored file's mimeType is intentionally left unchanged (new-version
        // semantics, not a type change) — only the content/revision is replaced.
        const requestBody: any = {};
        if (data.name) {
          requestBody.name = data.name;
        }
        file = await ctx.getDrive().files.update({
          fileId: data.fileId,
          requestBody,
          media: {
            mimeType: detectedMime,
            body: mediaBody()
          },
          fields: 'id, name, size, mimeType, webViewLink',
          supportsAllDrives: true
        });
      } else {
        const parentId = await ctx.resolveFolderId(data.parentFolderId);
        const requestBody: any = {
          name: uploadName,
          parents: [parentId]
        };
        if (targetMimeType) {
          requestBody.mimeType = targetMimeType;
        }
        file = await ctx.getDrive().files.create({
          requestBody,
          media: {
            mimeType: detectedMime,
            body: mediaBody()
          },
          fields: 'id, name, size, mimeType, webViewLink',
          supportsAllDrives: true
        });
      }

      ctx.log('File uploaded successfully', { fileId: file.data?.id, updated: !!data.fileId });
      return {
        content: [{
          type: "text",
          text: [
            data.fileId
              ? `Updated (new version): ${file.data?.name || fileName}`
              : `Uploaded: ${file.data?.name || fileName}`,
            `ID: ${file.data?.id || 'unknown'}`,
            `Size: ${file.data?.size || contentSize} bytes`,
            `Type: ${file.data?.mimeType || detectedMime}`,
            file.data?.webViewLink ? `Link: ${file.data.webViewLink}` : ''
          ].filter(Boolean).join('\n')
        }],
        isError: false
      };
    }

    case "downloadFile": {
      const validation = DownloadFileSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const data = validation.data;
      const downloadResult = await downloadDriveFile(ctx.getDrive(), data, ctx.log);

      return {
        content: [{
          type: 'text',
          text: [
            `Downloaded: ${downloadResult.driveName}`,
            `Saved to: ${downloadResult.resolvedPath}`,
            `Size: ${downloadResult.size} bytes`,
            downloadResult.isWorkspaceFile
              ? `Export format: ${downloadResult.exportMime}`
              : `Type: ${downloadResult.driveMimeType}`,
          ].join('\n'),
        }],
        isError: false,
      };
    }

    case "listPermissions": {
      const validation = ListPermissionsSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      const response = await ctx.getDrive().permissions.list({
        fileId: data.fileId,
        fields: 'permissions(id,type,role,emailAddress,domain,displayName,permissionDetails(inherited,inheritedFrom,permissionType))',
        supportsAllDrives: true,
      });

      const permissions = response.data.permissions || [];
      if (permissions.length === 0) {
        return { content: [{ type: 'text', text: 'No permissions found.' }], isError: false };
      }

      const lines = permissions.map((p) => {
        const who = p.emailAddress || p.domain || p.displayName || p.type || 'unknown';
        const inherited = p.permissionDetails?.some((d) => d.inherited === true) ?? false;
        const inheritedFrom = p.permissionDetails?.find((d) => d.inheritedFrom)?.inheritedFrom;
        const inheritedMarker = inherited
          ? ` [inherited${inheritedFrom ? ` from ${inheritedFrom}` : ''}]`
          : ' [direct]';
        return `- ${p.id}: ${who} (${p.type}) => ${p.role}${inheritedMarker}`;
      });

      return { content: [{ type: 'text', text: `Permissions for file ${data.fileId}:\n${lines.join('\n')}` }], isError: false };
    }

    case "addPermission": {
      const validation = AddPermissionSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      const response = await ctx.getDrive().permissions.create({
        fileId: data.fileId,
        requestBody: {
          type: data.type,
          role: data.role,
          ...((data.type === "user" || data.type === "group") && { emailAddress: data.emailAddress }),
          ...(data.type === "domain" && { domain: data.domain }),
          ...((data.type === "anyone" || data.type === "domain") && data.allowFileDiscovery !== undefined && { allowFileDiscovery: data.allowFileDiscovery }),
        },
        sendNotificationEmail: data.sendNotificationEmail,
        ...(data.emailMessage && { emailMessage: data.emailMessage }),
        fields: 'id,type,role,emailAddress,domain,allowFileDiscovery',
        supportsAllDrives: true,
      });

      const principal = response.data.emailAddress || response.data.domain || data.emailAddress || data.domain || data.type;
      const permissionId = response.data.id ?? 'unknown';
      const outcome = await reconcileCreatedPermissionRole(ctx.getDrive(), data.fileId, response.data, data.role);
      if (!outcome.ok) {
        return {
          content: [{
            type: 'text',
            text: `Permission MISMATCH for ${principal}: ${outcome.detail}. Verify with listPermissions and fix manually. Permission id: ${permissionId}`,
          }],
          isError: true,
        };
      }
      if (outcome.corrected) {
        return {
          content: [{
            type: 'text',
            text: `Permission added for ${principal}: Drive initially applied ${describeRole(outcome.initiallyApplied)} instead of the requested '${data.role}' (most likely because ${principal} already held a permission on this file); that permission's role was changed to '${outcome.role}'. Permission id: ${permissionId}`,
          }],
          isError: false,
        };
      }
      return { content: [{ type: 'text', text: `Permission added: ${permissionId} (${outcome.role}) for ${principal}` }], isError: false };
    }

    case "updatePermission": {
      const validation = UpdatePermissionSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      const response = await ctx.getDrive().permissions.update({
        fileId: data.fileId,
        permissionId: data.permissionId,
        requestBody: { role: data.role },
        fields: 'id,type,role,emailAddress',
        supportsAllDrives: true,
      });

      if (response.data.role !== data.role) {
        return {
          content: [{
            type: 'text',
            text: `Permission update MISMATCH: requested '${data.role}' for ${response.data.id ?? data.permissionId} but Drive reports ${describeRole(response.data.role)}. Verify with listPermissions.`,
          }],
          isError: true,
        };
      }
      return { content: [{ type: 'text', text: `Permission updated: ${response.data.id} => ${response.data.role}` }], isError: false };
    }

    case "removePermission": {
      const validation = RemovePermissionSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      let permissionId: string | undefined = data.permissionId;
      if (!permissionId && data.emailAddress) {
        const listed = await ctx.getDrive().permissions.list({
          fileId: data.fileId,
          fields: 'permissions(id,type,emailAddress)',
          supportsAllDrives: true,
        });
        const found = (listed.data.permissions || []).find(
          (p) => p.type === 'user' && (p.emailAddress || '').toLowerCase() === data.emailAddress!.toLowerCase(),
        );
        if (!found?.id) {
          return errorResponse(`No permission found for ${data.emailAddress}`);
        }
        permissionId = found.id;
      }

      if (!permissionId) {
        return errorResponse("Could not resolve a permission ID to remove");
      }

      await ctx.getDrive().permissions.delete({
        fileId: data.fileId,
        permissionId,
        supportsAllDrives: true,
      });

      return { content: [{ type: 'text', text: `Permission removed: ${permissionId}` }], isError: false };
    }

    case "shareFile": {
      const validation = ShareFileSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      // Idempotent behavior: update existing permission for the same principal instead of creating duplicates.
      const existing = await ctx.getDrive().permissions.list({
        fileId: data.fileId,
        fields: 'permissions(id,type,emailAddress,role)',
        supportsAllDrives: true,
      });

      const existingPerm = (existing.data.permissions || []).find(
        (p) => p.type === 'user' && (p.emailAddress || '').toLowerCase() === data.emailAddress.toLowerCase(),
      );

      if (existingPerm?.id) {
        if (existingPerm.role === data.role) {
          return {
            content: [{ type: 'text', text: `No changes needed: ${data.emailAddress} already has role ${data.role}. Permission ID: ${existingPerm.id}` }],
            isError: false,
          };
        }

        const updated = await ctx.getDrive().permissions.update({
          fileId: data.fileId,
          permissionId: existingPerm.id,
          requestBody: { role: data.role },
          fields: 'id,type,role,emailAddress',
          supportsAllDrives: true,
        });

        if (updated.data.role !== data.role) {
          return {
            content: [{
              type: 'text',
              text: `Permission update MISMATCH for ${updated.data.emailAddress || data.emailAddress}: requested '${data.role}' but Drive reports ${describeRole(updated.data.role)}. Verify with listPermissions. Permission ID: ${existingPerm.id}`,
            }],
            isError: true,
          };
        }

        return {
          content: [{ type: 'text', text: `Updated existing permission for ${updated.data.emailAddress || data.emailAddress} to ${updated.data.role}. Permission ID: ${updated.data.id}` }],
          isError: false,
        };
      }

      const response = await ctx.getDrive().permissions.create({
        fileId: data.fileId,
        requestBody: {
          type: 'user',
          role: data.role,
          emailAddress: data.emailAddress,
        },
        sendNotificationEmail: data.sendNotificationEmail,
        ...(data.emailMessage && { emailMessage: data.emailMessage }),
        fields: 'id,type,role,emailAddress',
        supportsAllDrives: true,
      });

      const recipient = response.data.emailAddress || data.emailAddress;
      const permissionId = response.data.id ?? 'unknown';
      const outcome = await reconcileCreatedPermissionRole(ctx.getDrive(), data.fileId, response.data, data.role);
      if (!outcome.ok) {
        return {
          content: [{
            type: 'text',
            text: `Permission MISMATCH for ${recipient}: ${outcome.detail}. Verify with listPermissions and fix manually. Permission ID: ${permissionId}`,
          }],
          isError: true,
        };
      }
      if (outcome.corrected) {
        return {
          content: [{
            type: 'text',
            text: `Shared file with ${recipient} as ${outcome.role}: Drive initially applied ${describeRole(outcome.initiallyApplied)} instead of the requested '${data.role}' (most likely because ${recipient} already held a permission on this file); that permission's role was changed to '${outcome.role}'. Permission ID: ${permissionId}`,
          }],
          isError: false,
        };
      }
      return {
        content: [{ type: 'text', text: `Shared file with ${recipient} as ${outcome.role}. Permission ID: ${permissionId}` }],
        isError: false,
      };
    }

    case "convertPdfToGoogleDoc": {
      const validation = ConvertPdfToGoogleDocSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      const source = await ctx.getDrive().files.get({
        fileId: data.fileId,
        fields: 'id,name,mimeType,parents',
        supportsAllDrives: true,
      });

      if (source.data.mimeType !== 'application/pdf') {
        return errorResponse(`File ${data.fileId} is not a PDF (mimeType=${source.data.mimeType || 'unknown'})`);
      }

      const parentId = data.parentFolderId || source.data.parents?.[0];
      const converted = await ctx.getDrive().files.copy({
        fileId: data.fileId,
        requestBody: {
          name: data.newName || `${source.data.name || 'Converted PDF'} (Doc)`,
          mimeType: 'application/vnd.google-apps.document',
          ...(parentId ? { parents: [parentId] } : {}),
        },
        fields: 'id,name,webViewLink,mimeType',
        supportsAllDrives: true,
      });

      return { content: [{ type: 'text', text: `Converted PDF to Google Doc: ${converted.data.name}\nID: ${converted.data.id}\nLink: ${converted.data.webViewLink}` }], isError: false };
    }

    case "bulkConvertFolderPdfs": {
      const validation = BulkConvertFolderPdfsSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      const list = await ctx.getDrive().files.list({
        q: `'${data.folderId}' in parents and mimeType='application/pdf' and trashed=false`,
        pageSize: data.maxResults,
        fields: 'files(id,name,mimeType)',
        // Parent-scoped: two flags only, no corpora=allDrives, so this PDF list
        // can never be an incompleteSearch partial result that silently skips
        // files before the conversion loop (#137).
        ...PARENT_SCOPED_LIST_PARAMS,
      });

      const files = list.data.files || [];
      const results: Array<{ id?: string; name?: string; docId?: string; ok: boolean; error?: string }> = [];

      // Sequential processing is intentional — parallel copies trigger Google API rate limits.
      for (const f of files) {
        try {
          const converted = await ctx.getDrive().files.copy({
            fileId: f.id!,
            requestBody: {
              name: `${f.name || 'Converted PDF'} (Doc)`,
              mimeType: 'application/vnd.google-apps.document',
              parents: [data.folderId],
            },
            fields: 'id,name',
            supportsAllDrives: true,
          });
          results.push({ id: f.id ?? undefined, name: f.name ?? undefined, docId: converted.data.id ?? undefined, ok: true });
        } catch (err: any) {
          const message = err?.message || 'Unknown conversion error';
          results.push({ id: f.id ?? undefined, name: f.name ?? undefined, ok: false, error: message });
          if (!data.continueOnError) break;
        }
      }

      const ok = results.filter(r => r.ok).length;
      const fail = results.length - ok;
      return {
        content: [{ type: 'text', text: `Bulk PDF conversion finished. Processed=${results.length}, Success=${ok}, Failed=${fail}\n\n${results.map(r => r.ok ? `✅ ${r.name} -> ${r.docId}` : `❌ ${r.name}: ${r.error}`).join('\n')}` }],
        isError: false,
      };
    }

    case "uploadPdfWithSplit": {
      const validation = UploadPdfWithSplitSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      if (!existsSync(data.localPath)) return errorResponse(`File not found: ${data.localPath}`);
      const parentId = await ctx.resolveFolderId(data.parentFolderId);

      if (!data.split) {
        const fileName = data.namePrefix || basename(data.localPath) || 'upload.pdf';
        const uploaded = await ctx.getDrive().files.create({
          requestBody: { name: fileName, parents: [parentId] },
          media: { mimeType: 'application/pdf', body: createReadStream(data.localPath) },
          fields: 'id,name,webViewLink',
          supportsAllDrives: true,
        });

        return {
          content: [{ type: 'text', text: `Uploaded PDF without split: ${uploaded.data.name}\nID: ${uploaded.data.id}` }],
          isError: false,
        };
      }

      const maxPagesPerChunk = data.maxPagesPerChunk ?? 25;
      const baseName = data.namePrefix || basename(data.localPath, extname(data.localPath));

      let tempDir: string | undefined;
      try {
        const splitResult = await splitPdfIntoChunkFiles(data.localPath, maxPagesPerChunk);
        tempDir = splitResult.tempDir;

        const uploadedParts: Array<{ id?: string | null; name?: string | null }> = [];
        for (let i = 0; i < splitResult.files.length; i++) {
          const partPath = splitResult.files[i];
          const partName = `${baseName}-part-${i + 1}.pdf`;

          const uploaded = await ctx.getDrive().files.create({
            requestBody: { name: partName, parents: [parentId] },
            media: { mimeType: 'application/pdf', body: createReadStream(partPath) },
            fields: 'id,name,webViewLink',
            supportsAllDrives: true,
          });

          uploadedParts.push({ id: uploaded.data.id, name: uploaded.data.name });
        }

        const lines = uploadedParts.map((p, idx) => `- part ${idx + 1}: ${p.name} (ID: ${p.id})`);
        return {
          content: [{
            type: 'text',
            text: `Uploaded split PDF into ${uploadedParts.length} part(s) using maxPagesPerChunk=${maxPagesPerChunk}\n${lines.join('\n')}`,
          }],
          isError: false,
        };
      } finally {
        if (tempDir) {
          await rm(tempDir, { recursive: true, force: true });
        }
      }
    }

    case "getRevisions": {
      const validation = GetRevisionsSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      const response = await ctx.getDrive().revisions.list({
        fileId: data.fileId,
        pageSize: data.pageSize,
        pageToken: data.pageToken,
        fields: 'nextPageToken,revisions(id,modifiedTime,lastModifyingUser(displayName,emailAddress),keepForever,size,originalFilename)',
      });

      const revisions: drive_v3.Schema$Revision[] = response.data.revisions || [];
      if (revisions.length === 0) {
        return { content: [{ type: 'text', text: `No revisions found for file ${data.fileId}.` }], isError: false };
      }

      const lines = revisions.map((r: drive_v3.Schema$Revision) => {
        const who = r.lastModifyingUser?.displayName || r.lastModifyingUser?.emailAddress || 'unknown';
        return `- ${r.id}: ${r.modifiedTime || 'unknown-time'} by ${who}${r.keepForever ? ' [kept]' : ''}`;
      });

      let text = `Revisions for file ${data.fileId}:\n${lines.join('\n')}`;
      if (response.data.nextPageToken) {
        text += `\n\nMore revisions available. Use pageToken="${response.data.nextPageToken}" to fetch the next page.`;
      }

      return {
        content: [{ type: 'text', text }],
        isError: false,
      };
    }

    case "restoreRevision": {
      const validation = RestoreRevisionSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      if (!data.confirm) {
        return errorResponse('Refusing restore: set confirm=true to restore a revision.');
      }

      try {
        // Get current file metadata to determine restore strategy
        const current = await ctx.getDrive().files.get({
          fileId: data.fileId,
          fields: 'name,mimeType',
          supportsAllDrives: true,
        });

        const fileMimeType = current.data.mimeType || '';
        const isWorkspaceFile = fileMimeType.startsWith('application/vnd.google-apps.');

        let revisionBody: unknown;
        let uploadMimeType: string;

        if (isWorkspaceFile) {
          // Workspace files don't support revisions.get with alt=media.
          // Use the revision's exportLinks to fetch content in an editable format.
          const revision = await ctx.getDrive().revisions.get({
            fileId: data.fileId,
            revisionId: data.revisionId,
            fields: 'id,exportLinks',
          });

          const exportLinks = (revision.data.exportLinks as Record<string, string> | null) || {};

          // Build preference list: editable formats from GOOGLE_WORKSPACE_EXPORT_FORMATS, excluding pdf
          const formatMap = GOOGLE_WORKSPACE_EXPORT_FORMATS[fileMimeType];
          const editableMimes = formatMap
            ? Object.entries(formatMap).filter(([ext]) => ext !== 'pdf').map(([, mime]) => mime)
            : [];

          // Pick the first editable MIME type available in exportLinks
          const selectedMime = editableMimes.find((m) => exportLinks[m])
            || Object.keys(exportLinks).find((m) => m !== 'application/pdf')
            || Object.keys(exportLinks)[0];

          if (!selectedMime || !exportLinks[selectedMime]) {
            return errorResponse('Selected revision has no usable export links for restore.');
          }

          uploadMimeType = selectedMime;

          // Fetch revision content from the export link using authenticated request
          const exportResponse = await ctx.authClient.request({ url: exportLinks[selectedMime], responseType: 'stream' });
          revisionBody = toNodeReadable(exportResponse.data);
        } else {
          // For binary files, download the revision content directly
          const revision = await ctx.getDrive().revisions.get(
            { fileId: data.fileId, revisionId: data.revisionId, alt: 'media' },
            { responseType: 'stream' },
          );
          revisionBody = toNodeReadable(revision.data);
          uploadMimeType = fileMimeType || 'application/octet-stream';
        }

        await ctx.getDrive().files.update({
          fileId: data.fileId,
          media: {
            mimeType: uploadMimeType,
            body: revisionBody,
          },
          supportsAllDrives: true,
        });

        const restoreMsg = `Restored file ${data.fileId} (${current.data.name || 'unnamed'}) from revision ${data.revisionId}.`;
        const workspaceWarning = isWorkspaceFile
          ? '\n\nWarning: This workspace file was restored via export/import. Some formatting or features (e.g. comments, suggestions, version history metadata) may have been lost.'
          : '';

        return {
          content: [{
            type: 'text',
            text: restoreMsg + workspaceWarning,
          }],
          isError: false,
        };
      } catch (err: unknown) {
        return errorResponse(`Failed to restore revision: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    case "authGetStatus": {
      const tokenPath = getSecureTokenPath();
      const tokenFileExists = existsSync(tokenPath);
      let scopeStatus: ReturnType<typeof resolveScopeStatus>;
      try {
        scopeStatus = resolveScopeStatus(ctx);
      } catch (e: unknown) {
        return errorResponse(`Invalid scope configuration: ${e instanceof Error ? e.message : String(e)}`);
      }
      const { requestedScopes, grantedScopes, missingScopes } = scopeStatus;
      const expiryDate = ctx.authClient?.credentials?.expiry_date as number | undefined;
      const expiresInSec = expiryDate ? Math.floor((expiryDate - Date.now()) / 1000) : null;

      // Which auth mode is actually active, and which override env vars are set.
      // Env-var *presence* forces service-account/external mode over tokens.json,
      // so a valid-looking token file can be silently bypassed (issue #137).
      const authMode = getActiveAuthMode();
      const envOverrides: Record<string, boolean> = {};
      // Mode-forcing vars come from the single source of truth so this list
      // can't drift from the modes authenticate() actually selects (#137).
      for (const varName of Object.values(AUTH_MODE_OVERRIDE_ENV_VARS)) {
        envOverrides[varName] = !!process.env[varName];
      }
      // These don't force a mode but change which tokens.json is read.
      envOverrides.GOOGLE_DRIVE_MCP_TOKEN_PATH = !!process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH;
      envOverrides.XDG_CONFIG_HOME = !!process.env.XDG_CONFIG_HOME;

      // Ask Google who the live Drive client actually is. This is the
      // load-bearing check for #137: a wrong/empty identity is otherwise
      // invisible (all calls return empty with no error). On the admin path
      // this tool tolerates a missing/unbuildable client (zero accounts, a
      // revoked grant), so getDrive() can throw — capture that as an
      // unresolved identity rather than crashing the very diagnostic meant to
      // explain it. getEffectiveIdentity never throws.
      let drive: drive_v3.Drive | undefined;
      try {
        drive = ctx.getDrive();
      } catch {
        drive = undefined;
      }
      const identity = await getEffectiveIdentity(drive);

      const warnings: string[] = [];
      if (authMode !== 'oauth') {
        const bypassWarning = describeBypassedTokens(authMode, tokenPath, tokenFileExists);
        if (bypassWarning) warnings.push(bypassWarning);
      }
      if (identity.error) {
        warnings.push(`Could not resolve the acting Google account via about.get: ${identity.error}`);
      } else if (!identity.emailAddress) {
        warnings.push('The acting Google account has no email — this is often an empty or misconfigured service account.');
      }

      const payload = {
        authMode,
        identity,
        tokenFilePath: tokenPath,
        tokenFileExists,
        envOverrides,
        hasAccessToken: !!ctx.authClient?.credentials?.access_token,
        hasRefreshToken: !!ctx.authClient?.credentials?.refresh_token,
        expiryDate: expiryDate || null,
        expiresInSec,
        requestedScopes,
        grantedScopes,
        missingScopes,
        warnings,
      };

      const status = computeAuthStatus({
        authMode,
        tokenFileExists,
        hasRefreshToken: payload.hasRefreshToken,
        grantedScopes,
        missingScopes,
        identityError: !!identity.error,
        warningCount: warnings.length,
      });

      const identityLine = identity.error
        ? `identity=UNRESOLVED (${identity.error})`
        : `identity=${identity.emailAddress || '(no email)'}`;

      let text = `Auth status (${status}):\n${JSON.stringify(payload, null, 2)}\n\nSummary: mode=${authMode}, ${identityLine}, token file ${tokenFileExists ? 'found' : 'missing'}, missing scopes=${missingScopes.length}.`;
      if (warnings.length > 0) {
        text += `\n\n⚠ ${warnings.join('\n⚠ ')}`;
      }
      if (grantedScopes.length === 0 && payload.hasAccessToken) {
        text += '\nNote: granted scopes may appear empty when the token was loaded from disk. This does not necessarily indicate missing permissions.';
      }

      return {
        content: [{ type: 'text', text }],
        isError: false,
      };
    }

    case "authListScopes": {
      let scopeStatus: ReturnType<typeof resolveScopeStatus>;
      try {
        scopeStatus = resolveScopeStatus(ctx);
      } catch (e: unknown) {
        return errorResponse(`Invalid scope configuration: ${e instanceof Error ? e.message : String(e)}`);
      }
      const { requestedScopes, grantedScopes, missingScopes } = scopeStatus;
      const presetsResolved = Object.fromEntries(
        Object.entries(SCOPE_PRESETS).map(([k, v]) => [k, v.map((s) => SCOPE_ALIASES[s] || s)]),
      );

      let text = `Scopes:\n${JSON.stringify({ requestedScopes, grantedScopes, missingScopes, presets: presetsResolved }, null, 2)}`;
      if (grantedScopes.length === 0 && !!ctx.authClient?.credentials?.access_token) {
        text += '\nNote: granted scopes may appear empty when the token was loaded from disk. This does not necessarily indicate missing permissions.';
      }

      return {
        content: [{ type: 'text', text }],
        isError: false,
      };
    }

    case "authTestFileAccess": {
      const validation = AuthTestFileAccessSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const data = validation.data;

      try {
        let check: { mode: string; [key: string]: unknown };
        if (data.fileId) {
          const file = await ctx.getDrive().files.get({
            fileId: data.fileId,
            fields: 'id,name,mimeType,permissions',
            supportsAllDrives: true,
          });
          check = { mode: 'file', fileId: file.data.id, name: file.data.name, mimeType: file.data.mimeType };
        } else {
          const list = await ctx.getDrive().files.list({
            pageSize: 1,
            fields: 'files(id,name,mimeType)',
            ...ALL_DRIVES_LIST_PARAMS,
          });
          check = { mode: 'list', visibleCount: list.data.files?.length || 0, sample: list.data.files?.[0] || null };
        }

        return {
          content: [{ type: 'text', text: `Auth access check OK:\n${JSON.stringify(check, null, 2)}` }],
          isError: false,
        };
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e);
        return {
          content: [{ type: 'text', text: `Auth access check failed:\n${JSON.stringify({ message }, null, 2)}` }],
          isError: true,
        };
      }
    }

    case "manage_accounts": {
      const validation = ManageAccountsSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const { action, account_id } = validation.data;

      try {
        switch (action) {
          case 'list': {
            const payload = {
              mode: ctx.accountOps.mode,
              defaultAccount: ctx.accountOps.getDefault() ?? null,
              accounts: ctx.accountOps.list(),
            };
            return {
              content: [{ type: 'text', text: JSON.stringify(payload, null, 2) }],
              isError: false,
            };
          }

          case 'add': {
            if (!account_id) return errorResponse(`action 'add' requires account_id (alias).`);
            const { authUrl, completion, cancel } = await ctx.accountOps.add(account_id);
            // Give the user up to 5 minutes to complete consent; on timeout, tear down
            // the embedded OAuth server so its callback port isn't leaked (finding 12).
            const record = await awaitConsentCompletion(
              { completion, cancel },
              ADD_ACCOUNT_CONSENT_TIMEOUT_MS,
            );
            const payload = {
              added: {
                alias: record.alias,
                email: record.email,
                sub: record.sub,
                pendingIdentity: !!record.pendingIdentity,
              },
              authUrl,
            };
            return {
              content: [{
                type: 'text',
                text: `Account added.\n${JSON.stringify(payload, null, 2)}`,
              }],
              isError: false,
            };
          }

          case 'remove': {
            if (!account_id) return errorResponse(`action 'remove' requires account_id.`);
            await ctx.accountOps.remove(account_id);
            return {
              content: [{ type: 'text', text: `Removed account '${account_id}'.` }],
              isError: false,
            };
          }

          case 'set_default': {
            const next = resolveSetDefaultTarget(account_id);
            await ctx.accountOps.setDefault(next);
            return {
              content: [{
                type: 'text',
                text: next === null ? 'Cleared default account.' : `Set default account to '${next}'.`,
              }],
              isError: false,
            };
          }
        }
      } catch (e: unknown) {
        return errorResponse(e instanceof Error ? e.message : String(e));
      }
      return null;
    }

    default:
      return null;
  }
}
