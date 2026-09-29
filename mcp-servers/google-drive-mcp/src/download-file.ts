import { createWriteStream, existsSync, renameSync, statSync, unlinkSync } from 'fs';
import { basename, dirname, extname, isAbsolute, join, relative, resolve } from 'path';
import { pipeline } from 'stream/promises';

export const GOOGLE_WORKSPACE_EXPORT_FORMATS: Record<string, Record<string, string>> = {
  'application/vnd.google-apps.document': {
    pdf: 'application/pdf',
    docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    md: 'text/markdown',
    txt: 'text/plain',
    html: 'text/html',
    rtf: 'application/rtf',
    odt: 'application/vnd.oasis.opendocument.text',
    epub: 'application/epub+zip',
  },
  'application/vnd.google-apps.spreadsheet': {
    xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    csv: 'text/csv',
    pdf: 'application/pdf',
    ods: 'application/vnd.oasis.opendocument.spreadsheet',
    tsv: 'text/tab-separated-values',
    html: 'text/html',
  },
  'application/vnd.google-apps.presentation': {
    pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    pdf: 'application/pdf',
    txt: 'text/plain',
    odp: 'application/vnd.oasis.opendocument.presentation',
  },
  'application/vnd.google-apps.drawing': {
    png: 'image/png',
    svg: 'image/svg+xml',
    pdf: 'application/pdf',
    jpg: 'image/jpeg',
  },
};

export const GOOGLE_WORKSPACE_DEFAULT_EXPORT: Record<string, { mimeType: string; ext: string }> = {
  'application/vnd.google-apps.document': { mimeType: 'application/pdf', ext: '.pdf' },
  'application/vnd.google-apps.spreadsheet': { mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', ext: '.xlsx' },
  'application/vnd.google-apps.presentation': { mimeType: 'application/vnd.openxmlformats-officedocument.presentationml.presentation', ext: '.pptx' },
  'application/vnd.google-apps.drawing': { mimeType: 'image/png', ext: '.png' },
};

export type DownloadFileArgs = {
  fileId: string;
  localPath: string;
  exportMimeType?: string;
  overwrite?: boolean;
};

export type DownloadFileResult = {
  driveName: string;
  driveMimeType: string;
  exportMime?: string;
  isWorkspaceFile: boolean;
  resolvedPath: string;
  size: number;
};

type DriveGetParams = {
  fileId: string;
  fields?: string;
  supportsAllDrives?: boolean;
  alt?: 'media';
};

type DriveExportParams = {
  fileId: string;
  mimeType?: string;
};

type DriveRequestOptions = {
  responseType?: 'stream';
};

type DriveResponse = {
  data: any;
};

type DriveLike = {
  files: {
    get: (params: DriveGetParams, options?: DriveRequestOptions) => Promise<DriveResponse>;
    export: (params: DriveExportParams, options?: DriveRequestOptions) => Promise<DriveResponse>;
  };
};

function sanitizeDriveFilename(driveName: string): string {
  return basename(driveName).replace(/^\.+/, '') || 'download';
}

function isPathWithinDirectory(targetPath: string, directoryPath: string): boolean {
  const relativePath = relative(resolve(directoryPath), resolve(targetPath));
  return relativePath === '' || (!relativePath.startsWith('..') && !isAbsolute(relativePath));
}

function resolveWorkspaceExport(
  driveMimeType: string,
  args: DownloadFileArgs,
  resolvedPath: string,
  isDirectory: boolean,
): { exportMime: string; fileExtForName: string } {
  const formatMap = GOOGLE_WORKSPACE_EXPORT_FORMATS[driveMimeType];
  if (!formatMap) {
    throw new Error(
      `Unsupported Google Workspace type for export: ${driveMimeType}. ` +
      'Supported types: Document, Spreadsheet, Presentation, Drawing.'
    );
  }

  if (args.exportMimeType) {
    const validMimes = Object.values(formatMap);
    if (!validMimes.includes(args.exportMimeType)) {
      throw new Error(
        `Unsupported export format '${args.exportMimeType}' for ${driveMimeType}. ` +
        `Supported: ${Object.entries(formatMap).map(([ext, mime]) => `${mime} (.${ext})`).join(', ')}`
      );
    }

    const extForMime = Object.entries(formatMap).find(([, mime]) => mime === args.exportMimeType)?.[0] || 'bin';
    return { exportMime: args.exportMimeType, fileExtForName: `.${extForMime}` };
  }

  if (!isDirectory && extname(resolvedPath)) {
    const ext = extname(resolvedPath).slice(1).toLowerCase();
    if (formatMap[ext]) {
      return { exportMime: formatMap[ext], fileExtForName: `.${ext}` };
    }
  }

  const defaultExport = GOOGLE_WORKSPACE_DEFAULT_EXPORT[driveMimeType];
  return { exportMime: defaultExport.mimeType, fileExtForName: defaultExport.ext };
}

function buildTempPath(resolvedPath: string): string {
  const random = Math.random().toString(16).slice(2);
  return `${resolvedPath}.download-${Date.now()}-${random}.tmp`;
}

export async function downloadDriveFile(
  drive: DriveLike,
  args: DownloadFileArgs,
  log: (message: string, data?: unknown) => void,
): Promise<DownloadFileResult> {
  if (!isAbsolute(args.localPath)) {
    throw new Error('localPath must be an absolute path');
  }

  const normalizedLocalPath = resolve(args.localPath);

  const fileMeta = await drive.files.get({
    fileId: args.fileId,
    fields: 'id, name, mimeType, size',
    supportsAllDrives: true,
  });

  const driveMimeType = fileMeta.data.mimeType;
  const driveName = fileMeta.data.name || 'download';

  if (!driveMimeType) {
    throw new Error('File has no MIME type');
  }

  const isWorkspaceFile = driveMimeType.startsWith('application/vnd.google-apps');
  const overwrite = args.overwrite ?? false;

  let resolvedPath = normalizedLocalPath;
  let isDirectory = false;

  if (existsSync(resolvedPath)) {
    isDirectory = statSync(resolvedPath).isDirectory();
  } else {
    const parentDir = dirname(resolvedPath);
    if (!existsSync(parentDir)) {
      throw new Error(`Parent directory does not exist: ${parentDir}`);
    }
  }

  let exportMime: string | undefined;
  let fileExtForName = '';

  if (isWorkspaceFile) {
    const exportSelection = resolveWorkspaceExport(driveMimeType, args, resolvedPath, isDirectory);
    exportMime = exportSelection.exportMime;
    fileExtForName = exportSelection.fileExtForName;
  }

  if (isDirectory) {
    const safeName = sanitizeDriveFilename(driveName);
    let fileName = safeName;
    if (isWorkspaceFile) {
      const nameWithoutExt = safeName.replace(/\.[^.]+$/, '');
      fileName = `${nameWithoutExt}${fileExtForName}`;
    }

    resolvedPath = join(resolvedPath, fileName);
    if (!isPathWithinDirectory(resolvedPath, normalizedLocalPath)) {
      throw new Error('Resolved file path escapes the target directory');
    }
  }

  const targetExists = existsSync(resolvedPath);
  if (targetExists && !overwrite) {
    throw new Error(`File already exists at ${resolvedPath}. Set overwrite: true to replace it.`);
  }

  log('Downloading file', {
    fileId: args.fileId,
    driveName,
    driveMimeType,
    isWorkspaceFile,
    exportMime,
    localPath: resolvedPath,
  });

  const response = isWorkspaceFile
    ? await drive.files.export({ fileId: args.fileId, mimeType: exportMime }, { responseType: 'stream' })
    : await drive.files.get({ fileId: args.fileId, alt: 'media', supportsAllDrives: true }, { responseType: 'stream' });

  const writePath = overwrite && targetExists ? buildTempPath(resolvedPath) : resolvedPath;
  const dest = createWriteStream(writePath);

  try {
    await pipeline(response.data, dest);
    if (writePath !== resolvedPath) {
      renameSync(writePath, resolvedPath);
    }
  } catch (downloadErr) {
    try {
      unlinkSync(writePath);
    } catch {
      // Ignore cleanup errors.
    }
    throw downloadErr;
  }

  const finalStats = statSync(resolvedPath);

  log('File downloaded successfully', {
    fileId: args.fileId,
    localPath: resolvedPath,
    size: finalStats.size,
  });

  return {
    driveName,
    driveMimeType,
    exportMime,
    isWorkspaceFile,
    resolvedPath,
    size: finalStats.size,
  };
}
