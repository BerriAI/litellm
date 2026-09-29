/**
 * Mock factories for all Google API services used by the MCP server.
 *
 * Each factory returns a service object that matches the shape expected by the
 * handler code, plus a `_tracker` for test assertions on call args.
 */

// ---------------------------------------------------------------------------
// Call tracker
// ---------------------------------------------------------------------------
export interface TrackedCall {
  method: string;
  args: any[];
}

export class CallTracker {
  calls: TrackedCall[] = [];
  record(method: string, args: any[]) {
    this.calls.push({ method, args });
  }
  reset() {
    this.calls = [];
  }
  getCalls(method: string) {
    return this.calls.filter((c) => c.method === method);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// The real Google API reads the upload's media.body stream. Mirror that here so a
// createReadStream (or any Readable) passed as media.body is fully consumed and closed
// before the call resolves — otherwise a stream over a temp fixture can open the file
// after the test deleted it, surfacing as async-activity-after-test-end.
async function drainMediaBody(args: any[]) {
  for (const a of args) {
    const body = a?.media?.body;
    if (body && typeof body.resume === 'function' && typeof body.on === 'function') {
      await new Promise<void>((resolve) => {
        body.on('end', resolve);
        body.on('error', resolve);
        body.resume();
      });
    }
  }
}

function stub(tracker: CallTracker, name: string, defaultReturn: any = {}) {
  let impl: ((...a: any[]) => any) | null = null;
  const fn = async (...args: any[]) => {
    tracker.record(name, args);
    await drainMediaBody(args);
    if (impl) return impl(...args);
    return { data: typeof defaultReturn === 'function' ? defaultReturn(...args) : defaultReturn };
  };
  fn._setImpl = (f: (...a: any[]) => any) => {
    impl = f;
  };
  fn._resetImpl = () => {
    impl = null;
  };
  return fn;
}

// ---------------------------------------------------------------------------
// Drive mock
// ---------------------------------------------------------------------------
export function createDriveMock() {
  const tracker = new CallTracker();
  const files = {
    list: stub(tracker, 'files.list', { files: [] }),
    create: stub(tracker, 'files.create', { id: 'file-1', name: 'test-file' }),
    get: stub(tracker, 'files.get', { id: 'file-1', name: 'test-file', mimeType: 'text/plain', parents: ['root'] }),
    update: stub(tracker, 'files.update', { id: 'file-1', name: 'test-file' }),
    delete: stub(tracker, 'files.delete', {}),
    copy: stub(tracker, 'files.copy', { id: 'file-copy-1', name: 'Copy of test-file', webViewLink: 'https://link' }),
    export: stub(tracker, 'files.export', {}),
  };
  const comments = {
    list: stub(tracker, 'comments.list', { comments: [] }),
    get: stub(tracker, 'comments.get', { id: 'comment-1', content: 'test comment', author: { displayName: 'User' } }),
    create: stub(tracker, 'comments.create', { id: 'comment-new', content: 'new comment' }),
    delete: stub(tracker, 'comments.delete', {}),
  };
  const drives = {
    list: stub(tracker, 'drives.list', { drives: [] }),
  };
  const about = {
    get: stub(tracker, 'about.get', {
      user: { displayName: 'Test User', emailAddress: 'test-user@example.com' },
      storageQuota: { limit: '16106127360', usage: '1024' },
    }),
  };
  const replies = {
    create: stub(tracker, 'replies.create', { id: 'reply-1', content: 'reply text' }),
  };
  const permissions = {
    // Echo the requested role and type: the handlers verify the applied role against
    // the request, so a fixed default would make every non-'reader' request look
    // like a Drive-side mismatch. Tests that want a mismatch override with _setImpl.
    create: stub(tracker, 'permissions.create', (req?: any) => ({
      id: 'perm-1', role: req?.requestBody?.role ?? 'reader', emailAddress: 'user@example.com', type: req?.requestBody?.type ?? 'user',
    })),
    list: stub(tracker, 'permissions.list', { permissions: [{ id: 'perm-1', role: 'reader', emailAddress: 'user@example.com', type: 'user' }] }),
    update: stub(tracker, 'permissions.update', (req?: any) => ({
      id: req?.permissionId ?? 'perm-1', role: req?.requestBody?.role ?? 'commenter', emailAddress: 'user@example.com', type: 'user',
    })),
    delete: stub(tracker, 'permissions.delete', {}),
    get: stub(tracker, 'permissions.get', { id: 'perm-1', role: 'reader', emailAddress: 'user@example.com', type: 'user' }),
  };
  const revisions = {
    list: stub(tracker, 'revisions.list', { revisions: [{ id: '1', modifiedTime: '2026-01-01T10:00:00Z', lastModifyingUser: { displayName: 'User' } }] }),
    get: stub(tracker, 'revisions.get', { id: '1', modifiedTime: '2026-01-01T10:00:00Z', exportLinks: { 'application/pdf': 'https://example.com/export.pdf' } }),
  };
  return { service: { files, comments, replies, permissions, revisions, drives, about }, tracker };
}

// ---------------------------------------------------------------------------
// Docs mock
// ---------------------------------------------------------------------------
export function createDocsMock() {
  const tracker = new CallTracker();
  const documents = {
    get: stub(tracker, 'documents.get', {
      documentId: 'doc-1',
      title: 'Test Doc',
      body: {
        content: [
          {
            paragraph: {
              elements: [
                { textRun: { content: 'Hello World\n' }, startIndex: 1, endIndex: 13 },
              ],
            },
            startIndex: 0,
            endIndex: 13,
          },
        ],
      },
    }),
    batchUpdate: stub(tracker, 'documents.batchUpdate', {}),
    create: stub(tracker, 'documents.create', { documentId: 'doc-new' }),
  };
  return { service: { documents }, tracker };
}

// ---------------------------------------------------------------------------
// Sheets mock
// ---------------------------------------------------------------------------
export function createSheetsMock() {
  const tracker = new CallTracker();
  const spreadsheets = {
    create: stub(tracker, 'spreadsheets.create', { spreadsheetId: 'sheet-1' }),
    get: stub(tracker, 'spreadsheets.get', {
      spreadsheetId: 'sheet-1',
      properties: { title: 'Test Sheet' },
      sheets: [{ properties: { sheetId: 0, title: 'Sheet1', gridProperties: { rowCount: 100, columnCount: 26 } } }],
    }),
    batchUpdate: stub(tracker, 'spreadsheets.batchUpdate', { replies: [{ addSheet: { properties: { title: 'Sheet2', sheetId: 1 } } }] }),
    values: {
      get: stub(tracker, 'spreadsheets.values.get', { values: [['a', 'b'], ['1', '2']] }),
      update: stub(tracker, 'spreadsheets.values.update', {}),
      append: stub(tracker, 'spreadsheets.values.append', { updates: { updatedCells: 4, updatedRows: 2, updatedRange: 'Sheet1!A1:B2' } }),
      batchUpdate: stub(tracker, 'spreadsheets.values.batchUpdate', {
        totalUpdatedCells: 4,
        responses: [{ updatedRange: 'Sheet1!A1:B2' }, { updatedRange: 'Sheet2!C1' }],
      }),
    },
  };
  return { service: { spreadsheets }, tracker };
}

// ---------------------------------------------------------------------------
// Slides mock
// ---------------------------------------------------------------------------
export function createSlidesMock() {
  const tracker = new CallTracker();
  const presentations = {
    create: stub(tracker, 'presentations.create', { presentationId: 'pres-1' }),
    get: stub(tracker, 'presentations.get', {
      presentationId: 'pres-1',
      slides: [
        {
          objectId: 'slide-1',
          pageElements: [
            { objectId: 'title-1', shape: { placeholder: { type: 'TITLE' }, text: { textElements: [{ textRun: { content: 'Title' } }] } } },
            { objectId: 'body-1', shape: { placeholder: { type: 'BODY' }, text: { textElements: [{ textRun: { content: 'Body' } }] } } },
          ],
          slideProperties: {
            notesPage: {
              objectId: 'notes-page-1',
              notesProperties: { speakerNotesObjectId: 'notes-1' },
              pageElements: [
                { objectId: 'notes-1', shape: { text: { textElements: [{ textRun: { content: 'Speaker notes text' } }] } } },
              ],
            },
          },
        },
      ],
    }),
    batchUpdate: stub(tracker, 'presentations.batchUpdate', { replies: [] }),
    pages: {
      get: stub(tracker, 'presentations.pages.get', {
        objectId: 'slide-1',
        pageElements: [
          { objectId: 'title-1', shape: { placeholder: { type: 'TITLE' } } },
          { objectId: 'body-1', shape: { placeholder: { type: 'BODY' } } },
        ],
      }),
      getThumbnail: stub(tracker, 'presentations.pages.getThumbnail', {
        contentUrl: 'https://slides.googleapis.com/mock-thumbnail.png',
      }),
    },
  };
  return { service: { presentations }, tracker };
}

// ---------------------------------------------------------------------------
// Calendar mock
// ---------------------------------------------------------------------------
export function createCalendarMock() {
  const tracker = new CallTracker();
  const calendarList = {
    list: stub(tracker, 'calendarList.list', {
      items: [{ id: 'primary', summary: 'My Calendar', primary: true, accessRole: 'owner' }],
    }),
  };
  const events = {
    list: stub(tracker, 'events.list', {
      items: [
        {
          id: 'event-1',
          summary: 'Test Event',
          start: { dateTime: '2025-01-01T10:00:00Z' },
          end: { dateTime: '2025-01-01T11:00:00Z' },
          status: 'confirmed',
          attachments: [
            {
              fileUrl: 'https://drive.google.com/file/d/file-1/view',
              title: 'Agenda.pdf',
              mimeType: 'application/pdf',
              fileId: 'file-1',
            },
          ],
        },
      ],
    }),
    get: stub(tracker, 'events.get', {
      id: 'event-1',
      summary: 'Test Event',
      start: { dateTime: '2025-01-01T10:00:00Z' },
      end: { dateTime: '2025-01-01T11:00:00Z' },
      status: 'confirmed',
      attachments: [
        {
          fileUrl: 'https://drive.google.com/file/d/file-1/view',
          title: 'Agenda.pdf',
          mimeType: 'application/pdf',
          fileId: 'file-1',
          iconLink: 'https://drive.google.com/icon.png',
        },
      ],
    }),
    insert: stub(tracker, 'events.insert', {
      id: 'event-new',
      summary: 'New Event',
      start: { dateTime: '2025-01-01T10:00:00Z' },
      end: { dateTime: '2025-01-01T11:00:00Z' },
      status: 'confirmed',
      htmlLink: 'https://calendar.google.com/event?id=event-new',
    }),
    update: stub(tracker, 'events.update', {
      id: 'event-1',
      summary: 'Updated Event',
      start: { dateTime: '2025-01-01T10:00:00Z' },
      end: { dateTime: '2025-01-01T11:00:00Z' },
      status: 'confirmed',
    }),
    delete: stub(tracker, 'events.delete', {}),
  };
  return { service: { calendarList, events }, tracker };
}

// ---------------------------------------------------------------------------
// Bundle all mocks together
// ---------------------------------------------------------------------------
export interface AllMocks {
  drive: ReturnType<typeof createDriveMock>;
  docs: ReturnType<typeof createDocsMock>;
  sheets: ReturnType<typeof createSheetsMock>;
  slides: ReturnType<typeof createSlidesMock>;
  calendar: ReturnType<typeof createCalendarMock>;
  google: Record<string, (...args: any[]) => any>;
}

export function createAllMocks(): AllMocks {
  const drive = createDriveMock();
  const docs = createDocsMock();
  const sheets = createSheetsMock();
  const slides = createSlidesMock();
  const calendar = createCalendarMock();

  const google: Record<string, (...args: any[]) => any> = {
    drive: () => drive.service,
    docs: () => docs.service,
    sheets: () => sheets.service,
    slides: () => slides.service,
    calendar: () => calendar.service,
  };

  return { drive, docs, sheets, slides, calendar, google };
}
