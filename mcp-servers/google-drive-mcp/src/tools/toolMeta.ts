// ---------------------------------------------------------------------------
// Per-tool metadata: opKind (read/write/admin) and acceptable OAuth scopes.
//
// Consumed by the dispatch layer in src/index.ts to (a) inject the optional
// `account` parameter into non-admin tool schemas and (b) resolve which
// account(s) a tool call should target via the AccountResolver.
//
// Any-of scope semantics: an account is eligible for a tool when it has
// granted at least ONE of the acceptable scopes. For a given tool we list
// every Google scope that would allow it to run, from narrowest to broadest.
// ---------------------------------------------------------------------------

import { ToolOpKind } from '../auth/types.js';

export interface ToolMeta {
  opKind: ToolOpKind;
  acceptableScopes: string[];
}

// -- Scope constants --------------------------------------------------------

const DRIVE_READONLY = 'https://www.googleapis.com/auth/drive.readonly';
const DRIVE_FILE = 'https://www.googleapis.com/auth/drive.file';
const DRIVE = 'https://www.googleapis.com/auth/drive';
const DOCUMENTS = 'https://www.googleapis.com/auth/documents';
const SPREADSHEETS = 'https://www.googleapis.com/auth/spreadsheets';
const PRESENTATIONS = 'https://www.googleapis.com/auth/presentations';
const CALENDAR = 'https://www.googleapis.com/auth/calendar';
const CALENDAR_EVENTS = 'https://www.googleapis.com/auth/calendar.events';
const CALENDAR_READONLY = 'https://www.googleapis.com/auth/calendar.readonly';

const DRIVE_READ_SCOPES = [DRIVE_READONLY, DRIVE_FILE, DRIVE];
const DRIVE_WRITE_SCOPES = [DRIVE_FILE, DRIVE];
// Docs/Sheets/Slides APIs also accept the drive-family scopes, so an account
// consented with `drive` (or drive.file/drive.readonly) is eligible too. Split
// read/write because drive.readonly can authorize reads but not writes.
const DOCS_READ_SCOPES = [DOCUMENTS, DRIVE_READONLY, DRIVE_FILE, DRIVE];
const DOCS_WRITE_SCOPES = [DOCUMENTS, DRIVE_FILE, DRIVE];
const SHEETS_READ_SCOPES = [SPREADSHEETS, DRIVE_READONLY, DRIVE_FILE, DRIVE];
const SHEETS_WRITE_SCOPES = [SPREADSHEETS, DRIVE_FILE, DRIVE];
const SLIDES_READ_SCOPES = [PRESENTATIONS, DRIVE_READONLY, DRIVE_FILE, DRIVE];
const SLIDES_WRITE_SCOPES = [PRESENTATIONS, DRIVE_FILE, DRIVE];
const CAL_READ_SCOPES = [CALENDAR_READONLY, CALENDAR, CALENDAR_EVENTS];
const CAL_WRITE_SCOPES = [CALENDAR, CALENDAR_EVENTS];

// -- Helpers ----------------------------------------------------------------

const read = (scopes: string[]): ToolMeta => ({ opKind: 'read', acceptableScopes: scopes });
const write = (scopes: string[]): ToolMeta => ({ opKind: 'write', acceptableScopes: scopes });
const admin: ToolMeta = { opKind: 'admin', acceptableScopes: [] };

// -- Registry ---------------------------------------------------------------

export const TOOL_META: Record<string, ToolMeta> = {
  // ---- Drive ----
  search: read(DRIVE_READ_SCOPES),
  listFolder: read(DRIVE_READ_SCOPES),
  listSharedDrives: read(DRIVE_READ_SCOPES),
  downloadFile: read(DRIVE_READ_SCOPES),
  listPermissions: read(DRIVE_READ_SCOPES),
  getRevisions: read(DRIVE_READ_SCOPES),

  readTextFile: read(DRIVE_READ_SCOPES),

  createTextFile: write(DRIVE_WRITE_SCOPES),
  updateTextFile: write(DRIVE_WRITE_SCOPES),
  createFolder: write(DRIVE_WRITE_SCOPES),
  deleteItem: write(DRIVE_WRITE_SCOPES),
  renameItem: write(DRIVE_WRITE_SCOPES),
  moveItem: write(DRIVE_WRITE_SCOPES),
  copyFile: write(DRIVE_WRITE_SCOPES),
  uploadFile: write(DRIVE_WRITE_SCOPES),
  addPermission: write(DRIVE_WRITE_SCOPES),
  updatePermission: write(DRIVE_WRITE_SCOPES),
  removePermission: write(DRIVE_WRITE_SCOPES),
  shareFile: write(DRIVE_WRITE_SCOPES),
  convertPdfToGoogleDoc: write(DRIVE_WRITE_SCOPES),
  bulkConvertFolderPdfs: write(DRIVE_WRITE_SCOPES),
  uploadPdfWithSplit: write(DRIVE_WRITE_SCOPES),
  restoreRevision: write(DRIVE_WRITE_SCOPES),
  createShortcut: write(DRIVE_WRITE_SCOPES),
  lockFile: write(DRIVE_WRITE_SCOPES),
  unlockFile: write(DRIVE_WRITE_SCOPES),

  // ---- Docs ----
  readGoogleDoc: read(DOCS_READ_SCOPES),
  readGoogleDocPaginated: read(DOCS_READ_SCOPES),
  listDocumentTabs: read(DOCS_READ_SCOPES),
  listComments: read(DRIVE_READ_SCOPES),
  getComment: read(DRIVE_READ_SCOPES),
  getGoogleDocContent: read(DOCS_READ_SCOPES),
  getGoogleDocContentPaginated: read(DOCS_READ_SCOPES),
  getGoogleDocImage: read(DOCS_READ_SCOPES),
  listGoogleDocs: read(DRIVE_READ_SCOPES),
  getDocumentInfo: read(DOCS_READ_SCOPES),
  getGoogleDocStyleSummary: read(DOCS_READ_SCOPES),
  describeGoogleDocRange: read(DOCS_READ_SCOPES),
  readSmartChips: read(DOCS_READ_SCOPES),

  createGoogleDoc: write(DOCS_WRITE_SCOPES),
  // createDocFromHTML uploads via drive.files.create (HTML → Doc conversion),
  // so it needs a drive-family write scope, not the docs scope.
  createDocFromHTML: write(DRIVE_WRITE_SCOPES),
  updateGoogleDoc: write(DOCS_WRITE_SCOPES),
  insertText: write(DOCS_WRITE_SCOPES),
  deleteRange: write(DOCS_WRITE_SCOPES),
  applyTextStyle: write(DOCS_WRITE_SCOPES),
  applyParagraphStyle: write(DOCS_WRITE_SCOPES),
  formatGoogleDocText: write(DOCS_WRITE_SCOPES),
  formatGoogleDocParagraph: write(DOCS_WRITE_SCOPES),
  createParagraphBullets: write(DOCS_WRITE_SCOPES),
  findAndReplaceInDoc: write(DOCS_WRITE_SCOPES),
  addComment: write(DRIVE_WRITE_SCOPES),
  replyToComment: write(DRIVE_WRITE_SCOPES),
  deleteComment: write(DRIVE_WRITE_SCOPES),
  insertTable: write(DOCS_WRITE_SCOPES),
  editTableCell: write(DOCS_WRITE_SCOPES),
  styleDocTable: write(DOCS_WRITE_SCOPES),
  insertImageFromUrl: write(DOCS_WRITE_SCOPES),
  insertLocalImage: write(DOCS_WRITE_SCOPES),
  addDocumentTab: write(DOCS_WRITE_SCOPES),
  renameDocumentTab: write(DOCS_WRITE_SCOPES),
  insertSmartChip: write(DOCS_WRITE_SCOPES),
  createFootnote: write(DOCS_WRITE_SCOPES),

  // ---- Sheets ----
  getGoogleSheetContent: read(SHEETS_READ_SCOPES),
  getGoogleSheetCells: read(SHEETS_READ_SCOPES),
  getSpreadsheetInfo: read(SHEETS_READ_SCOPES),
  listSheets: read(SHEETS_READ_SCOPES),
  listGoogleSheets: read(DRIVE_READ_SCOPES),

  createGoogleSheet: write(SHEETS_WRITE_SCOPES),
  updateGoogleSheet: write(SHEETS_WRITE_SCOPES),
  batchUpdateGoogleSheetValues: write(SHEETS_WRITE_SCOPES),
  updateGoogleSheetIfUnchanged: write(SHEETS_WRITE_SCOPES),
  formatGoogleSheetCells: write(SHEETS_WRITE_SCOPES),
  formatGoogleSheetText: write(SHEETS_WRITE_SCOPES),
  formatGoogleSheetNumbers: write(SHEETS_WRITE_SCOPES),
  setGoogleSheetBorders: write(SHEETS_WRITE_SCOPES),
  mergeGoogleSheetCells: write(SHEETS_WRITE_SCOPES),
  addGoogleSheetConditionalFormat: write(SHEETS_WRITE_SCOPES),
  appendSpreadsheetRows: write(SHEETS_WRITE_SCOPES),
  addSpreadsheetSheet: write(SHEETS_WRITE_SCOPES),
  addSheet: write(SHEETS_WRITE_SCOPES),
  renameSheet: write(SHEETS_WRITE_SCOPES),
  deleteSheet: write(SHEETS_WRITE_SCOPES),
  addDataValidation: write(SHEETS_WRITE_SCOPES),
  protectRange: write(SHEETS_WRITE_SCOPES),
  addNamedRange: write(SHEETS_WRITE_SCOPES),
  setColumnWidth: write(SHEETS_WRITE_SCOPES),
  setRowHeight: write(SHEETS_WRITE_SCOPES),
  autoResizeColumns: write(SHEETS_WRITE_SCOPES),
  autoResizeRows: write(SHEETS_WRITE_SCOPES),
  hideSheetDimension: write(SHEETS_WRITE_SCOPES),
  showSheetDimension: write(SHEETS_WRITE_SCOPES),
  addDimensionGroup: write(SHEETS_WRITE_SCOPES),
  deleteDimensionGroup: write(SHEETS_WRITE_SCOPES),
  updateDimensionGroup: write(SHEETS_WRITE_SCOPES),
  listDimensionGroups: read(SHEETS_READ_SCOPES),

  // ---- Slides ----
  getGoogleSlidesContent: read(SLIDES_READ_SCOPES),
  getGoogleSlidesSpeakerNotes: read(SLIDES_READ_SCOPES),
  getSlideElementInfo: read(SLIDES_READ_SCOPES),
  exportSlideThumbnail: read(SLIDES_READ_SCOPES),

  createGoogleSlides: write(SLIDES_WRITE_SCOPES),
  updateGoogleSlides: write(SLIDES_WRITE_SCOPES),
  formatGoogleSlidesText: write(SLIDES_WRITE_SCOPES),
  formatGoogleSlidesParagraph: write(SLIDES_WRITE_SCOPES),
  styleGoogleSlidesShape: write(SLIDES_WRITE_SCOPES),
  setGoogleSlidesBackground: write(SLIDES_WRITE_SCOPES),
  createGoogleSlidesTextBox: write(SLIDES_WRITE_SCOPES),
  createGoogleSlidesShape: write(SLIDES_WRITE_SCOPES),
  updateGoogleSlidesSpeakerNotes: write(SLIDES_WRITE_SCOPES),
  deleteGoogleSlide: write(SLIDES_WRITE_SCOPES),
  duplicateSlide: write(SLIDES_WRITE_SCOPES),
  reorderSlides: write(SLIDES_WRITE_SCOPES),
  replaceAllTextInSlides: write(SLIDES_WRITE_SCOPES),
  insertSlidesImageFromUrl: write(SLIDES_WRITE_SCOPES),
  moveSlideElement: write(SLIDES_WRITE_SCOPES),
  deleteSlideElement: write(SLIDES_WRITE_SCOPES),
  insertSlidesLocalImage: write(SLIDES_WRITE_SCOPES),
  setSlideVisibility: write(SLIDES_WRITE_SCOPES),
  replaceSlideImage: write(SLIDES_WRITE_SCOPES),
  setElementZOrder: write(SLIDES_WRITE_SCOPES),
  setElementText: write(SLIDES_WRITE_SCOPES),

  // ---- Calendar ----
  listCalendars: read(CAL_READ_SCOPES),
  getCalendarEvents: read(CAL_READ_SCOPES),
  getCalendarEvent: read(CAL_READ_SCOPES),
  createCalendarEvent: write(CAL_WRITE_SCOPES),
  updateCalendarEvent: write(CAL_WRITE_SCOPES),
  deleteCalendarEvent: write(CAL_WRITE_SCOPES),

  // ---- Admin ----
  manage_accounts: admin,
  authGetStatus: admin,
  authListScopes: admin,
  authTestFileAccess: admin,
};

/** Tool names that bypass account resolution and always run on the default account. */
export const ADMIN_TOOLS: ReadonlySet<string> = new Set(
  Object.entries(TOOL_META)
    .filter(([, m]) => m.opKind === 'admin')
    .map(([name]) => name),
);

/** Default meta for an unrecognized tool name — treated as a read with no scope filter. */
export const FALLBACK_META: ToolMeta = { opKind: 'read', acceptableScopes: [] };
