import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import test from 'node:test';

import { downloadDriveFile } from '../src/download-file.js';

type MockDriveOptions = {
  mimeType: string;
  name: string;
  streamFactory?: () => Readable | ReadableStream<Uint8Array>;
};

function createMockDrive(options: MockDriveOptions) {
  const calls = {
    metadataGetCount: 0,
    mediaGetCount: 0,
    exportCount: 0,
    lastExportMime: undefined as string | undefined,
  };

  const drive = {
    files: {
      get: async (params: { alt?: string }) => {
        if (params.alt === 'media') {
          calls.mediaGetCount += 1;
          return { data: options.streamFactory ? options.streamFactory() : Readable.from(['payload']) };
        }

        calls.metadataGetCount += 1;
        return {
          data: {
            id: 'file-123',
            name: options.name,
            mimeType: options.mimeType,
            size: '7',
          },
        };
      },
      export: async (params: { mimeType?: string }) => {
        calls.exportCount += 1;
        calls.lastExportMime = params.mimeType;
        return { data: options.streamFactory ? options.streamFactory() : Readable.from(['payload']) };
      },
    },
  };

  return { drive, calls };
}

function createTempDir(): string {
  return mkdtempSync(join(tmpdir(), 'download-file-test-'));
}

function createFailingStream(message: string): Readable {
  let sent = false;
  return new Readable({
    read() {
      if (!sent) {
        sent = true;
        this.push('partial-data');
        this.destroy(new Error(message));
      }
    },
  });
}

test('rejects relative localPath before hitting Drive API', async () => {
  const { drive, calls } = createMockDrive({
    mimeType: 'text/plain',
    name: 'note.txt',
  });

  await assert.rejects(
    () => downloadDriveFile(drive, { fileId: 'file-123', localPath: 'relative/path.txt' }, () => {}),
    /absolute path/
  );

  assert.equal(calls.metadataGetCount, 0);
});

test('sanitizes malicious Drive filename when localPath is a directory', async () => {
  const tempDir = createTempDir();
  try {
    const { drive } = createMockDrive({
      mimeType: 'text/plain',
      name: '../../etc/passwd',
    });

    const result = await downloadDriveFile(
      drive,
      { fileId: 'file-123', localPath: tempDir },
      () => {}
    );

    const expectedPath = join(tempDir, 'passwd');
    assert.equal(result.resolvedPath, expectedPath);
    assert.equal(readFileSync(expectedPath, 'utf8'), 'payload');
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test('uses extension-based export MIME for Workspace files', async () => {
  const tempDir = createTempDir();
  try {
    const outputPath = join(tempDir, 'sheet.csv');
    const { drive, calls } = createMockDrive({
      mimeType: 'application/vnd.google-apps.spreadsheet',
      name: 'Quarterly Plan',
    });

    const result = await downloadDriveFile(
      drive,
      { fileId: 'file-123', localPath: outputPath },
      () => {}
    );

    assert.equal(calls.exportCount, 1);
    assert.equal(calls.mediaGetCount, 0);
    assert.equal(calls.lastExportMime, 'text/csv');
    assert.equal(result.exportMime, 'text/csv');
    assert.equal(readFileSync(outputPath, 'utf8'), 'payload');
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test('downloads a media response that is a web ReadableStream (gaxios 7 shape)', async () => {
  const tempDir = createTempDir();
  try {
    const outputPath = join(tempDir, 'web-stream.txt');
    const { drive } = createMockDrive({
      mimeType: 'text/plain',
      name: 'web-stream.txt',
      streamFactory: () =>
        new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('web-payload'));
            controller.close();
          },
        }),
    });

    await downloadDriveFile(
      drive,
      { fileId: 'file-123', localPath: outputPath },
      () => {}
    );

    assert.equal(readFileSync(outputPath, 'utf8'), 'web-payload');
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test('returns an error when overwrite is false and target file exists', async () => {
  const tempDir = createTempDir();
  try {
    const outputPath = join(tempDir, 'existing.txt');
    writeFileSync(outputPath, 'original-content', 'utf8');

    const { drive } = createMockDrive({
      mimeType: 'text/plain',
      name: 'ignored.txt',
    });

    await assert.rejects(
      () =>
        downloadDriveFile(
          drive,
          { fileId: 'file-123', localPath: outputPath, overwrite: false },
          () => {}
        ),
      /already exists/
    );

    assert.equal(readFileSync(outputPath, 'utf8'), 'original-content');
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test('preserves original file on failed overwrite and removes temporary partial file', async () => {
  const tempDir = createTempDir();
  try {
    const outputPath = join(tempDir, 'existing.txt');
    writeFileSync(outputPath, 'original-content', 'utf8');

    const { drive } = createMockDrive({
      mimeType: 'text/plain',
      name: 'existing.txt',
      streamFactory: () => createFailingStream('network interrupted'),
    });

    await assert.rejects(
      () =>
        downloadDriveFile(
          drive,
          { fileId: 'file-123', localPath: outputPath, overwrite: true },
          () => {}
        ),
      /network interrupted/
    );

    assert.equal(readFileSync(outputPath, 'utf8'), 'original-content');
    const leftoverTmpFiles = readdirSync(tempDir).filter(
      (name) => name.includes('.download-') && name.endsWith('.tmp')
    );
    assert.deepEqual(leftoverTmpFiles, []);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test('replaces existing file when overwrite is true and download succeeds', async () => {
  const tempDir = createTempDir();
  try {
    const outputPath = join(tempDir, 'existing.txt');
    writeFileSync(outputPath, 'old-content', 'utf8');

    const { drive } = createMockDrive({
      mimeType: 'text/plain',
      name: 'existing.txt',
      streamFactory: () => Readable.from(['new-content']),
    });

    await downloadDriveFile(
      drive,
      { fileId: 'file-123', localPath: outputPath, overwrite: true },
      () => {}
    );

    assert.equal(existsSync(outputPath), true);
    assert.equal(readFileSync(outputPath, 'utf8'), 'new-content');
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});
