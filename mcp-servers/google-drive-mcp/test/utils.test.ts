import assert from 'node:assert/strict';
import test from 'node:test';

import {
  getExtensionFromFilename,
  getMimeTypeFromFilename,
  escapeDriveQuery,
  parseA1Range,
  colToIndex,
  convertA1ToGridRange,
  buildCalendarEventUpdate,
} from '../src/utils.js';

// ---------------------------------------------------------------------------
// getExtensionFromFilename
// ---------------------------------------------------------------------------
test('getExtensionFromFilename returns lowercase extension', () => {
  assert.equal(getExtensionFromFilename('report.PDF'), 'pdf');
});

test('getExtensionFromFilename returns last extension for dotted names', () => {
  assert.equal(getExtensionFromFilename('archive.tar.gz'), 'gz');
});

test('getExtensionFromFilename returns empty string for no extension', () => {
  assert.equal(getExtensionFromFilename('Makefile'), 'makefile');
});

// ---------------------------------------------------------------------------
// getMimeTypeFromFilename
// ---------------------------------------------------------------------------
test('getMimeTypeFromFilename returns text/plain for .txt', () => {
  assert.equal(getMimeTypeFromFilename('notes.txt'), 'text/plain');
});

test('getMimeTypeFromFilename returns text/markdown for .md', () => {
  assert.equal(getMimeTypeFromFilename('README.md'), 'text/markdown');
});

test('getMimeTypeFromFilename falls back to text/plain for unknown', () => {
  assert.equal(getMimeTypeFromFilename('data.csv'), 'text/plain');
});

// ---------------------------------------------------------------------------
// escapeDriveQuery
// ---------------------------------------------------------------------------
test('escapeDriveQuery passes clean strings through', () => {
  assert.equal(escapeDriveQuery('hello world'), 'hello world');
});

test('escapeDriveQuery escapes single quotes', () => {
  assert.equal(escapeDriveQuery("it's"), "it\\'s");
});

test('escapeDriveQuery escapes backslashes', () => {
  assert.equal(escapeDriveQuery('path\\file'), 'path\\\\file');
});

test('escapeDriveQuery escapes backslash before quote', () => {
  assert.equal(escapeDriveQuery("a\\'b"), "a\\\\\\'b");
});

// ---------------------------------------------------------------------------
// parseA1Range
// ---------------------------------------------------------------------------
test('parseA1Range defaults to Sheet1 when no prefix', () => {
  const result = parseA1Range('A1:B2');
  assert.equal(result.sheetName, 'Sheet1');
  assert.equal(result.cellRange, 'A1:B2');
});

test('parseA1Range extracts unquoted sheet name', () => {
  const result = parseA1Range('Data!C3:D4');
  assert.equal(result.sheetName, 'Data');
  assert.equal(result.cellRange, 'C3:D4');
});

test('parseA1Range strips surrounding single quotes from sheet name', () => {
  const result = parseA1Range("'My Sheet'!A1:B2");
  assert.equal(result.sheetName, 'My Sheet');
  assert.equal(result.cellRange, 'A1:B2');
});

test('parseA1Range handles sheet name with embedded quote', () => {
  // Google Sheets escapes embedded quotes by doubling: 'Sheet''s Name'
  const result = parseA1Range("'Sheet''s Name'!A1");
  assert.equal(result.sheetName, "Sheet''s Name");
  assert.equal(result.cellRange, 'A1');
});

test('parseA1Range handles single cell without sheet prefix', () => {
  const result = parseA1Range('Z99');
  assert.equal(result.sheetName, 'Sheet1');
  assert.equal(result.cellRange, 'Z99');
});

// ---------------------------------------------------------------------------
// colToIndex
// ---------------------------------------------------------------------------
test('colToIndex converts A to 0', () => {
  assert.equal(colToIndex('A'), 0);
});

test('colToIndex converts Z to 25', () => {
  assert.equal(colToIndex('Z'), 25);
});

test('colToIndex converts AA to 26', () => {
  assert.equal(colToIndex('AA'), 26);
});

test('colToIndex converts AZ to 51', () => {
  assert.equal(colToIndex('AZ'), 51);
});

test('colToIndex converts BA to 52', () => {
  assert.equal(colToIndex('BA'), 52);
});

// ---------------------------------------------------------------------------
// convertA1ToGridRange
// ---------------------------------------------------------------------------
test('convertA1ToGridRange parses single cell A1', () => {
  const result = convertA1ToGridRange('A1', 0);
  assert.deepEqual(result, {
    sheetId: 0,
    startColumnIndex: 0,
    startRowIndex: 0,
    endColumnIndex: 1,
    endRowIndex: 1,
  });
});

test('convertA1ToGridRange parses range B2:D5', () => {
  const result = convertA1ToGridRange('B2:D5', 3);
  assert.deepEqual(result, {
    sheetId: 3,
    startColumnIndex: 1,
    startRowIndex: 1,
    endColumnIndex: 4,
    endRowIndex: 5,
  });
});

test('convertA1ToGridRange parses full-column range A:C', () => {
  const result = convertA1ToGridRange('A:C', 0);
  assert.deepEqual(result, {
    sheetId: 0,
    startColumnIndex: 0,
    endColumnIndex: 3,
  });
});

test('convertA1ToGridRange parses full-row range 1:3', () => {
  const result = convertA1ToGridRange('1:3', 0);
  assert.deepEqual(result, {
    sheetId: 0,
    startRowIndex: 0,
    endRowIndex: 3,
  });
});

test('convertA1ToGridRange throws on invalid notation', () => {
  assert.throws(() => convertA1ToGridRange('invalid!', 0), /Invalid A1 notation/);
});

// ---------------------------------------------------------------------------
// buildCalendarEventUpdate
// ---------------------------------------------------------------------------
const EXISTING_EVENT = {
  id: 'event-1',
  kind: 'calendar#event',
  etag: '"abc"',
  htmlLink: 'https://calendar.google.com/event',
  iCalUID: 'uid@google.com',
  creator: { email: 'creator@test.com' },
  organizer: { email: 'organizer@test.com' },
  sequence: 3,
  summary: 'Original Title',
  description: 'Original Description',
  location: 'Room A',
  start: { dateTime: '2025-01-01T10:00:00Z' },
  end: { dateTime: '2025-01-01T11:00:00Z' },
  attendees: [{ email: 'alice@test.com' }, { email: 'bob@test.com' }],
  recurrence: ['RRULE:FREQ=WEEKLY'],
  visibility: 'default',
  reminders: { useDefault: true },
};

test('buildCalendarEventUpdate preserves existing values when no overrides', () => {
  const result = buildCalendarEventUpdate(EXISTING_EVENT, {});
  assert.equal(result.summary, 'Original Title');
  assert.equal(result.description, 'Original Description');
  assert.equal(result.location, 'Room A');
  assert.deepEqual(result.start, EXISTING_EVENT.start);
  assert.deepEqual(result.end, EXISTING_EVENT.end);
  assert.deepEqual(result.attendees, EXISTING_EVENT.attendees);
  assert.deepEqual(result.recurrence, EXISTING_EVENT.recurrence);
  assert.equal(result.visibility, 'default');
  assert.deepEqual(result.reminders, { useDefault: true });
});

test('buildCalendarEventUpdate user overrides win', () => {
  const result = buildCalendarEventUpdate(EXISTING_EVENT, {
    summary: 'New Title',
    description: 'New Description',
    location: 'Room B',
  });
  assert.equal(result.summary, 'New Title');
  assert.equal(result.description, 'New Description');
  assert.equal(result.location, 'Room B');
  // Unchanged fields preserved
  assert.deepEqual(result.start, EXISTING_EVENT.start);
  assert.deepEqual(result.attendees, EXISTING_EVENT.attendees);
});

test('buildCalendarEventUpdate allows empty string overrides', () => {
  const result = buildCalendarEventUpdate(EXISTING_EVENT, {
    summary: '',
    description: '',
    location: '',
  });
  assert.equal(result.summary, '');
  assert.equal(result.description, '');
  assert.equal(result.location, '');
});

test('buildCalendarEventUpdate maps attendees from string[] to {email}[]', () => {
  const result = buildCalendarEventUpdate(EXISTING_EVENT, {
    attendees: ['charlie@test.com', 'diana@test.com'],
  });
  assert.deepEqual(result.attendees, [
    { email: 'charlie@test.com' },
    { email: 'diana@test.com' },
  ]);
});

test('buildCalendarEventUpdate allows empty attendees array', () => {
  const result = buildCalendarEventUpdate(EXISTING_EVENT, {
    attendees: [],
  });
  assert.deepEqual(result.attendees, []);
});

test('buildCalendarEventUpdate excludes read-only fields', () => {
  const result = buildCalendarEventUpdate(EXISTING_EVENT, { summary: 'X' });
  assert.equal(result.id, undefined);
  assert.equal(result.kind, undefined);
  assert.equal(result.etag, undefined);
  assert.equal(result.htmlLink, undefined);
  assert.equal(result.iCalUID, undefined);
  assert.equal(result.creator, undefined);
  assert.equal(result.organizer, undefined);
  assert.equal(result.sequence, undefined);
});

test('buildCalendarEventUpdate overrides start/end', () => {
  const newStart = { dateTime: '2025-06-01T09:00:00Z' };
  const newEnd = { dateTime: '2025-06-01T10:00:00Z' };
  const result = buildCalendarEventUpdate(EXISTING_EVENT, { start: newStart, end: newEnd });
  assert.deepEqual(result.start, newStart);
  assert.deepEqual(result.end, newEnd);
});

test('buildCalendarEventUpdate preserves recurrence from existing', () => {
  const result = buildCalendarEventUpdate(EXISTING_EVENT, { summary: 'Changed' });
  assert.deepEqual(result.recurrence, ['RRULE:FREQ=WEEKLY']);
});

test('buildCalendarEventUpdate preserves existing attachments when not overridden', () => {
  const withAttachments = {
    ...EXISTING_EVENT,
    attachments: [{ fileUrl: 'https://drive.google.com/f/1', title: 'A', fileId: '1' }],
  };
  const result = buildCalendarEventUpdate(withAttachments, { summary: 'Changed' });
  assert.deepEqual(result.attachments, withAttachments.attachments);
});

test('buildCalendarEventUpdate replaces attachments when overridden', () => {
  const withAttachments = {
    ...EXISTING_EVENT,
    attachments: [{ fileUrl: 'https://drive.google.com/f/1', title: 'A' }],
  };
  const result = buildCalendarEventUpdate(withAttachments, {
    attachments: [{ fileUrl: 'https://drive.google.com/f/2', title: 'B' }],
  });
  assert.deepEqual(result.attachments, [{ fileUrl: 'https://drive.google.com/f/2', title: 'B' }]);
});

test('buildCalendarEventUpdate clears attachments when given an empty array', () => {
  const withAttachments = {
    ...EXISTING_EVENT,
    attachments: [{ fileUrl: 'https://drive.google.com/f/1', title: 'A' }],
  };
  const result = buildCalendarEventUpdate(withAttachments, { attachments: [] });
  assert.deepEqual(result.attachments, []);
});
