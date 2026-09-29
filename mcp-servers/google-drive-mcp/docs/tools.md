# Tool reference

This server exposes 130 MCP tools across Google Drive, Docs, Sheets, Slides, and Calendar. Tool availability can depend on the granted OAuth scopes. Unless noted otherwise, every tool also accepts the optional top-level `account` parameter described in [Authentication](authentication.md#per-tool-account-selection).

## Available Tools

### Search and Navigation
- **search** - Search for files across Google Drive
  - `query`: Search terms (or raw Drive API query when `rawQuery=true`)
  - `pageSize`: Number of results per page (optional, default 50, max 100)
  - `pageToken`: Pagination token for next page (optional)
  - `rawQuery`: Pass `query` directly to the Drive API — enables operators like `modifiedTime`, `createdTime`, `mimeType`, `name contains`, etc. (optional)
  - `orderBy`: Sort order — `modifiedTime desc` (default), `modifiedTime`, `createdTime desc`, `createdTime`, `recency desc`, `recency`, `name`, or `name_natural`. Keys without `desc` sort ascending, so plain `modifiedTime` is oldest-first. When paging with `pageToken`, pass the same `orderBy` on every call so the ordering stays consistent across pages — the response repeats the effective value alongside the token (optional)

- **listFolder** - List contents of a folder
  - `folderId`: Folder ID (optional, defaults to root)
  - `pageSize`: Number of results (optional, max 100)
  - `pageToken`: Pagination token (optional)

- **listSharedDrives** - List available Google Shared Drives
  - `pageSize`: Number of drives to return (optional, default 50, max 100)
  - `pageToken`: Pagination token (optional)

### File Management
- **createTextFile** - Create a text or markdown file
  - `name`: File name (must end with .txt or .md)
  - `content`: File content
  - `parentFolderId`: Parent folder ID (optional)

- **updateTextFile** - Update existing text file
  - `fileId`: File ID to update
  - `content`: New content
  - `name`: New name (optional)

- **readTextFile** - Read any `text/*` Drive file; use `readGoogleDoc` for native Google Docs
  - `fileId`: File ID
  - `maxLength`: Maximum Unicode code points to return before truncation (optional)

- **deleteItem** - Move a file or folder to trash (not a permanent deletion - items can be restored from Google Drive trash)
  - `itemId`: Item ID to move to trash

- **renameItem** - Rename a file or folder
  - `itemId`: Item ID to rename
  - `newName`: New name

- **moveItem** - Move a file or folder
  - `itemId`: Item ID to move
  - `destinationFolderId`: Destination folder ID

- **copyFile** - Create a copy of a Google Drive file or document
  - `fileId`: ID of the file to copy
  - `newName`: Name for the copied file (optional, defaults to "Copy of [original name]")
  - `parentFolderId`: Destination folder ID (optional, defaults to same location)

- **createShortcut** - Create a Drive shortcut without duplicating the target
  - `targetFileId`: Target file or folder ID
  - `parentFolderId`: Destination folder ID or path (optional)
  - `shortcutName`: Custom shortcut name (optional)

- **lockFile** - Prevent edits by adding a content restriction
  - `fileId`: File ID
  - `reason`: Reason shown to editors (optional)
  - `ownerRestricted`: Allow only the owner to unlock (optional, default: false)

- **unlockFile** - Remove a file's content restriction
  - `fileId`: File ID

#### Sharing and Permissions
- **listPermissions** - List current sharing permissions on a file/folder
  - `fileId`: File or folder ID

- **addPermission** - Add a new permission to a file/folder. If the principal already holds a permission, Drive updates it instead of creating a second one, and the requested role is applied even when it is a downgrade
  - `fileId`: File or folder ID
  - `type`: Permission target type (`user`, `group`, `domain`, `anyone`)
  - `role`: Permission role (`reader`, `commenter`, `writer`, `fileOrganizer`, `organizer`, `owner`)
  - `emailAddress`: Required for `user`/`group` types
  - `domain`: Required for `domain` type
  - `allowFileDiscovery`: For `domain`/`anyone` only — `false` (default) = accessible with the link, `true` = discoverable in search (optional)
  - `sendNotificationEmail`: Send notification email (optional)

- **updatePermission** - Update role for an existing permission
  - `fileId`: File or folder ID
  - `permissionId`: Permission ID
  - `role`: New role

- **removePermission** - Remove a permission from a file/folder
  - `fileId`: File or folder ID
  - `permissionId`: Permission ID (optional if `emailAddress` is provided)
  - `emailAddress`: Email to find permission by (optional fallback)

- **shareFile** - Share file with a user email (idempotent helper). An existing permission for that user is updated to the requested role, even when it is a downgrade, instead of creating a duplicate
  - `fileId`: File or folder ID
  - `emailAddress`: Recipient email
  - `role`: Role (`reader`, `commenter`, `writer`)
  - `sendNotificationEmail`: Send notification email (optional)

#### File Revisions (v1.7.0)
- **getRevisions** - List revisions for a file
  - `fileId`: File ID
  - `pageSize`: Max revisions to return (optional)

- **restoreRevision** - Restore a file from a selected revision (safety-confirmed)
  - `fileId`: File ID
  - `revisionId`: Revision ID to restore
  - `confirm`: Must be `true` to execute restore

#### Auth Diagnostics (v1.7.0)
- **authGetStatus** - Show token/scopes/auth health diagnostics (machine + human readable). Reports the **active auth mode** (`oauth`/`service_account`/`external_token`) and the **effective Google identity** the live Drive client is actually acting as (via Drive `about.get`), and warns when an environment variable is causing your `tokens.json` to be ignored
- **authListScopes** - Show configured/requested scopes, granted scopes, missing scopes, and presets
- **authTestFileAccess** - Test Drive access (optionally against a specific `fileId`)

- **uploadFile** - Upload a file (any type: image, audio, video, PDF, etc.) to Google Drive, either from a local path or from base64-encoded content. Can also upload the content as a new version of an existing file (in-place update)
  - `localPath`: Absolute path to the local file (provide either `localPath` or `contentBase64`)
  - `contentBase64`: Base64-encoded file content — alternative to `localPath`, useful for remote/HTTP deployments where the client has no access to the server's filesystem (optional). Must be valid (standard) base64; invalid input is rejected
  - `fileId`: ID of an existing file to update in place — the uploaded content becomes a new version of that file, keeping its ID, links, and revision history (optional; omit to create a new file. Not combinable with `parentFolderId` or `convertToGoogleFormat`)
  - `name`: File name in Drive (optional, defaults to local filename; required when creating a new file from `contentBase64`)
  - `parentFolderId`: Parent folder ID or path (optional, e.g., '/Work/Projects')
  - `mimeType`: MIME type (optional, auto-detected from extension)
  - `convertToGoogleFormat`: Convert uploaded file to native Google Workspace format (optional, default: false). When enabled, Office files are automatically converted:
    - `.docx` / `.doc` → Google Doc
    - `.xlsx` / `.xls` → Google Sheet
    - `.pptx` / `.ppt` → Google Slides
    - File extension is stripped from the name automatically (e.g., `report.docx` becomes `report`)

- **downloadFile** - Download a Google Drive file to a local path
  - `fileId`: Google Drive file ID
  - `localPath`: Absolute local path to save the file (can be a directory or full file path)
  - `exportMimeType`: For Google Workspace files, MIME type to export as (optional, e.g., 'application/pdf', 'text/csv')
  - `overwrite`: Whether to overwrite existing files (optional, default: false)

#### PDF Ingestion and Conversion (v1.6.0)
- **convertPdfToGoogleDoc** - Convert a PDF already stored in Drive into an editable Google Doc
  - `fileId`: Source PDF file ID
  - `newName`: Optional destination doc name
  - `parentFolderId`: Optional destination folder

- **bulkConvertFolderPdfs** - Convert all PDFs in a folder and return per-file success/failure summary
  - `folderId`: Source folder ID
  - `maxResults`: Maximum PDFs to process (optional, default: 100)
  - `continueOnError`: Continue processing after individual failures (optional, default: true)

- **uploadPdfWithSplit** - Upload a local PDF, optionally split into chunked PDF parts before upload
  - `localPath`: Absolute local path to PDF
  - `split`: Enable split mode metadata output (optional, default: false)
  - `maxPagesPerChunk`: Advisory chunk size for split planning (optional)
  - `parentFolderId`: Optional destination folder
  - `namePrefix`: Optional uploaded file name prefix

### Folder Operations
- **createFolder** - Create a new folder
  - `name`: Folder name
  - `parent`: Parent folder ID or path (optional)

### Google Docs

<a id="ifrevisionid"></a>
**Optimistic locking with `ifRevisionId`.** Every Google Docs write tool accepts an
optional `ifRevisionId`. Pass the `revisionId` reported by `readGoogleDoc`,
`getGoogleDocContent`, `getGoogleDocContentPaginated`, or `getDocumentInfo`, and the
write is rejected by the API if the document changed since that read — so an agent
edit cannot silently clobber a concurrent human edit. Omit it and the write applies
unconditionally, as before.

Three limits to know:

- A `revisionId` is only guaranteed valid for **24 hours**.
- It **cannot be shared across users**. In multi-account mode a `revisionId` read
  under one `manage_accounts` alias will not lock a write issued under another, so
  read and write under the same account.
- Docs populates it **only for callers with edit access**. A read-only caller sees
  `revisionId: unavailable (no edit access)` instead of a value.

Note that the Drive `version` integer and the revision ids from `getRevisions` are
different identifiers; neither is accepted as a lock.

#### Create and Update
- **createGoogleDoc** - Create a Google Doc
  - `name`: Document name
  - `content`: Document content
  - `parentFolderId`: Parent folder ID (optional)

- **createDocFromHTML** - Create a Google Doc from HTML with styles, lists, tables, links, and inline images
  - `name`: Document name
  - `html`: HTML content
  - `parentFolderId`: Parent folder ID (optional)

- **updateGoogleDoc** - Replace all content in a Google Doc. The replacement is one atomic batchUpdate, so a failure leaves the document unchanged rather than wiped. Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - `content`: New content

#### Reading and Discovery
- **readGoogleDoc** - Read content of a Google Doc with format options. `text`/`markdown` output leads with a `revisionId:` line (counted against `maxLength`); `json` output is the raw document, which already carries the field
  - `documentId`: Document ID
  - `format`: Output format — `text`, `json`, or `markdown` (optional, default: text)
  - `maxLength`: Maximum characters to return (optional)
  - Inline images are surfaced (not dropped): `markdown` renders `![alt](contentUri "objectId=…")`; `text` emits a single-line `[image: objectId=… contentUri=… sourceUri=… size=WxHpt]` token. Pass the `objectId` to `getGoogleDocImage` to fetch the bytes. Floating/anchored (positioned) images are not surfaced.

- **readGoogleDocPaginated** - Read a large Google Doc one page at a time (avoids host output-size truncation)
  - `documentId`: Document ID
  - `format`: Output format — `text` or `markdown` (optional, default: text)
  - `offset`: Character offset into the output text (optional, default: 0; pass the previous response's `nextOffset`)
  - `limit`: Maximum characters per page (optional, default: 50000, max: 80000)
  - `tabId`: Read a specific tab by ID (optional)

- **getGoogleDocContent** - Get document content with text indices for formatting. Trails with `Total length` and a `revisionId:` line, so the indices and the lock come from the same fetch
  - `documentId`: Document ID
  - `includeFormatting`: Include font, style, color, and baseline (superscript/subscript) info for each text span, plus one `¶` meta line per paragraph with non-default paragraph styles (named style, alignment, visible borders, shading) over its real index span (optional, default: false)
  - Inline images render as a single-line `[image: objectId=… contentUri=… sourceUri=… size=WxHpt]` token (was a bare `[image]`). Pass the `objectId` to `getGoogleDocImage`.
  - Tables emit ONE real `[start-end]` span for the whole table, then the pipe rendering, then a `cells: r0c0 [a-b], r0c1 [c-d]` map of each cell's real index range. Row lines carry no ranges of their own — deriving them from the rendered markdown produced numbers that ran past the table's end into the content after it. Use the cell ranges for `applyTextStyle` / `formatGoogleDocText` inside a cell, and the table's own `startIndex` with `editTableCell` for whole-cell edits
  - On a multi-tab document the tab header prints its id (`=== Tab: Name (tabId=…) ===`) and the table hint names that id, because index spaces restart per tab and `editTableCell` searches only the first tab when given no `tabId`

- **getGoogleDocContentPaginated** - Paginated `getGoogleDocContent`; page ends snap to a line boundary where possible (a single line longer than `limit` is hard-cut to make forward progress). A table is kept whole: its `<table …>` header, pipe rows, and `cells:` map move to the next page together rather than being split. A table longer than `limit` still takes a hard cut, but never between the header and its first row. The JSON result carries a `revisionId` field
  - `documentId`: Document ID
  - `includeFormatting`: Include font, style, color, and baseline (superscript/subscript) info for each text span, plus one `¶` meta line per paragraph with non-default paragraph styles (named style, alignment, visible borders, shading) over its real index span (optional, default: false)
  - `offset`: Character offset into the formatted output (optional, default: 0; pass the previous response's `nextOffset`)
  - `limit`: Maximum characters per page (optional, default: 50000, max: 80000)
  - Because page ends snap to a line boundary, a page can end on a `¶` meta line whose paragraph text starts the next page. The index prefix on both lines makes them pairable, so no extra call is needed to reunite them.

- **getGoogleDocImage** - Fetch the bytes of an inline image embedded in a Google Doc, keyed by its inline object ID (the doc is re-fetched so the underlying image URL is always fresh). Inline images only; floating/anchored (positioned) images are not supported.
  - `documentId`: Document ID
  - `inlineObjectId`: The inline object ID shown in `readGoogleDoc` / `getGoogleDocContent` image placeholders (the `objectId=…` value, e.g. `kix.abc123`)
  - `outputFormat`: `image` (default) returns a native image the model can view; `base64` returns a `{ inlineObjectId, mimeType, byteLength, dataBase64 }` JSON envelope for programmatic use (forwarding, save-to-disk)

- **listDocumentTabs** - List all tabs in a Google Doc with their IDs and hierarchy
  - `documentId`: Document ID
  - `includeContent`: Include content summary (character count) for each tab (optional)

- **addDocumentTab** - Add a new tab in a Google Doc
  - `documentId`: Document ID
  - `title`: Tab title

- **renameDocumentTab** - Rename an existing tab in a Google Doc
  - `documentId`: Document ID
  - `tabId`: Tab ID
  - `title`: New tab title

- **insertSmartChip** - Insert a person smart chip (mention) at a document index. Only person chips are supported by the Docs API; date and file chips are read-only. Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - `index`: Insertion index (1-based)
  - `chipType`: `person` (only supported type)
  - `personEmail`: Email address for the person mention

- **readSmartChips** - Read smart chip-like elements (person mentions, rich links, date chips) from the default tab of a document. Only the default tab is scanned; other tabs are not included.
  - `documentId`: Document ID

- **createFootnote** - Create a footnote in a Google Doc. Footnotes cannot be inserted inside equations, headers, footers, or other footnotes. Accepts [`ifRevisionId`](#ifrevisionid) on the creating call
  - `documentId`: Document ID
  - `index`: 1-based character index where the footnote reference should be inserted (optional — provide this or `endOfSegment`)
  - `endOfSegment`: If true, insert footnote at the end of the document body (optional — provide this or `index`)
  - `content`: Optional text content for the footnote body

- **listGoogleDocs** - List Google Documents with optional filtering
  - `query`: Search query to filter by name or content (optional)
  - `maxResults`: Maximum documents to return, 1-100 (optional, default: 20)
  - `orderBy`: Sort order — `modifiedTime desc` (default), `modifiedTime`, `createdTime desc`, `createdTime`, `recency desc`, `recency`, `name`, or `name_natural`. Keys without `desc` sort ascending, so plain `modifiedTime` is oldest-first (optional)

- **getDocumentInfo** - Get detailed metadata about a specific Google Document, including the Docs `revisionId` (distinct from the Drive `version` counter also shown)
  - `documentId`: Document ID

- **getGoogleDocStyleSummary** - Compact whole-document style inventory in a few hundred tokens: fonts with size ladders, named-style counts, text colors, bordered and shaded paragraph locations, a table inventory with real index ranges, and the heading outline. The cheap first read before formatting work, in place of a full document JSON fetch. Reports the `revisionId` for use as [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - `tabId`: Summarize only this tab, from `listDocumentTabs` (optional; omit to cover every tab)
  - Counts and the font inventory aggregate across tabs; locations are prefixed with the tab title on a multi-tab document, because index spaces restart in each tab
  - Lists longer than 12 entries (20 for the outline) are truncated with an explicit `… (+N more)`, so a clipped list is never mistaken for a short one

- **describeGoogleDocRange** - Style probe for one spot: paragraph styles (named style, alignment, borders, shading, indents, spacing, bullets) and text runs (font, size, bold/italic/underline, colors, links) overlapping a range, each with its real document index so the result feeds straight into `applyParagraphStyle` or `applyTextStyle`. Reports the `revisionId` for use as [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - `startIndex`: Start index (1-based) — use this or `textToFind`
  - `endIndex`: End index (exclusive; defaults to `startIndex` + 1). Only valid alongside `startIndex`, and must be greater than it
  - `textToFind`: Probe the styling at this exact text, case-sensitive — use this or `startIndex`. Mixing the two targeting modes is refused rather than silently resolved
  - `matchInstance`: Which occurrence of `textToFind` (1-based, optional, default 1)
  - `tabId`: Tab to probe, from `listDocumentTabs` (optional; defaults to the first tab)
  - Describes one tab's body only. Headers, footers, and footnotes each start their own index space at 0, so including them would report several paragraphs with overlapping ranges and no way to tell which an index-taking write should target

#### Surgical Editing
- **insertText** - Insert text at an index or relative to found text (doesn't replace entire doc). Accepts [`ifRevisionId`](#ifrevisionid) on Google Docs; it is refused on text files rather than ignored
  - `documentId`: Document ID
  - `text`: Text to insert
  - Target (use one): `index` (1-based position) OR `textToFind`+`matchInstance` with `position` (`before`/`after`, default `after`; Google Docs only; exact, case-sensitive match). A match ending at the document's final paragraph break inserts `after` just before that break

- **deleteRange** - Delete an index range or found text. Accepts [`ifRevisionId`](#ifrevisionid) on Google Docs; it is refused on text files rather than ignored
  - `documentId`: Document ID
  - Target (use one): `startIndex` (1-based, inclusive) + `endIndex` (exclusive) OR `textToFind`+`matchInstance` (Google Docs only; exact, case-sensitive match). A match ending at the document's final paragraph break is trimmed to keep that break

#### Text and Paragraph Styling
- **applyTextStyle** - Apply text formatting (bold, italic, color, etc.) to a range or found text. Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - Target (use one): `startIndex`+`endIndex` OR `textToFind`+`matchInstance`
  - `bold`, `italic`, `underline`, `strikethrough`: Text styling (optional)
  - `fontSize`: Font size in points (optional)
  - `fontFamily`: Font family name (optional)
  - `foregroundColor`: Hex color, e.g., `#FF0000` (optional)
  - `backgroundColor`: Hex background color (optional)
  - `linkUrl`: URL for hyperlink (optional)
  - `baselineOffset`: `SUPERSCRIPT`, `SUBSCRIPT`, or `NONE` to reset to the normal baseline (optional)

- **applyParagraphStyle** - Apply paragraph formatting, including borders and shading. Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - Target (use one): `startIndex`+`endIndex` OR `textToFind`+`matchInstance` OR `indexWithinParagraph`
  - `namedStyleType`: NORMAL_TEXT, TITLE, SUBTITLE, HEADING_1 through HEADING_6 (optional)
  - `alignment`: START, CENTER, END, or JUSTIFIED (optional)
  - `indentStart`, `indentEnd`, `indentFirstLine`: Indents in points (optional)
  - `spaceAbove`, `spaceBelow`: Spacing in points (optional)
  - `keepWithNext`, `keepLinesTogether`, `avoidWidowAndOrphan`, `pageBreakBefore`: Pagination controls (optional)
  - `borderTop`, `borderBottom`, `borderLeft`, `borderRight`, `borderBetween`: Paragraph borders, each `{ color?, width?, padding?, dashStyle? }` — hex color (default `#000000`), width/padding in points (both default 1), `SOLID`/`DOT`/`DASH` (default `SOLID`) (optional)
  - `removeBorders`: Edges to clear at the paragraph level — `top`, `bottom`, `left`, `right`, `between`, or `all`. Each cleared edge falls back to the border its named style defines (usually none); `horizontalRule` elements are not affected. The update styles every paragraph overlapping the range, so `removeBorders: ["all"]` over the whole body strips every paragraph border in one call (optional)
  - `shading`: Paragraph background color as hex, e.g., `#F1F3F4`; `removeShading`: clear it (optional)

- **formatGoogleDocText** - Alias for `applyTextStyle` (compatibility helper)
  - Same parameters as `applyTextStyle`

- **formatGoogleDocParagraph** - Alias for `applyParagraphStyle` (compatibility helper)
  - Same parameters as `applyParagraphStyle`

#### Bullet Points and Lists
- **createParagraphBullets** - Add or remove bullet points / numbered lists on paragraphs. Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - Target (use one): `startIndex`+`endIndex` OR `textToFind`+`matchInstance`
  - `bulletPreset`: Bullet style preset (optional, default: `BULLET_DISC_CIRCLE_SQUARE`). Available presets:
    - **Bullet styles**: `BULLET_DISC_CIRCLE_SQUARE`, `BULLET_DIAMONDX_ARROW3D_SQUARE`, `BULLET_CHECKBOX`, `BULLET_ARROW_DIAMOND_DISC`, `BULLET_STAR_CIRCLE_SQUARE`, `BULLET_ARROW3D_CIRCLE_SQUARE`, `BULLET_LEFTTRIANGLE_DIAMOND_DISC`
    - **Numbered styles**: `NUMBERED_DECIMAL_ALPHA_ROMAN`, `NUMBERED_DECIMAL_ALPHA_ROMAN_PARENS`, `NUMBERED_DECIMAL_NESTED`, `NUMBERED_UPPERALPHA_ALPHA_ROMAN`, `NUMBERED_UPPERROMAN_UPPERALPHA_DECIMAL`, `NUMBERED_ZERODECIMAL_ALPHA_ROMAN`
    - **Remove bullets**: `NONE` — removes existing bullets/numbering from the targeted paragraphs

- **findAndReplaceInDoc** - Find and replace text across a Google Doc. On zero matches, the response names the likeliest lookalike cause (non-breaking spaces, curly quotes, `&amp;` entities, case). A `replaceText` containing `\n` is compiled to exact delete+insert requests in one atomic batch (real paragraph breaks) instead of the API's `replaceAllText`, which does not render an embedded newline as a paragraph break — the surrounding paragraphs flatten into one and characters are dropped. `findText` must stay single-line, since a match cannot span a paragraph break. Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - `findText`: Text to find
  - `replaceText`: Replacement text
  - `matchCase`: Case-sensitive match (optional, default: false)
  - `dryRun`: Only count matches (exact — body, tables, headers, footers, footnotes), don’t modify the document (optional, default: false)
  - `expectedCount`: Safety guard — the exact number of occurrences you expect. Counted before writing; a mismatch aborts without modifying the document (optional). Costs one extra document read per call, and a zero-match result costs one more to diagnose the likeliest cause

#### Tables and Images
- **insertTable** - Insert a new table at a given index. Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - `rows`: Number of rows
  - `columns`: Number of columns
  - `index`: Position to insert at (1-based)

- **editTableCell** - Edit content and/or style of a specific table cell. Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - `tableStartIndex`: Starting index of the table element
  - `rowIndex`: Row index (0-based)
  - `columnIndex`: Column index (0-based)
  - `textContent`: New text content (optional)
  - `bold`, `italic`, `fontSize`, `alignment`: Cell styling (optional)

- **styleDocTable** - Style a Docs table in one atomic call (cell borders, background, padding, content alignment, column widths, row heights). Accepts [`ifRevisionId`](#ifrevisionid)
  - `documentId`: Document ID
  - `tableStartIndex`: Start index of the TABLE element — take it from the table's span line in `readGoogleDoc` / `getGoogleDocContent`, which prints `tableStartIndex=N` (and `tabId=` on a multi-tab document). Never compute it from text offsets
  - Cell subset targeting (optional; default whole table): `rowIndex`+`columnIndex` (0-based) with `rowSpan`/`columnSpan` (default 1)
  - `backgroundColor`: Cell background as hex; `removeBackground`: clear it (optional)
  - `contentAlignment`: `TOP`, `MIDDLE`, or `BOTTOM` (optional)
  - `padding`: Cell padding on all four edges, in points (optional)
  - `borderTop`, `borderBottom`, `borderLeft`, `borderRight`: Cell borders, each `{ color?, width?, dashStyle? }` (optional)
  - `removeBorders`: Edges to clear via width-0 borders (`top`, `bottom`, `left`, `right`, or `all`) — `["all"]` unboxes the table in one call (optional)
  - `columnWidth`: Fixed column width in points (min 5), scoped by `columnIndices` (0-based; omit for all columns) (optional)
  - `minRowHeight`: Minimum row height in points, scoped by `rowIndices` (0-based; omit for all rows) (optional)

- **insertImageFromUrl** - Insert an inline image from a publicly accessible URL
  - `documentId`: Document ID
  - `imageUrl`: Publicly accessible URL to the image
  - `index`: Position to insert at (1-based)
  - `width`, `height`: Image dimensions in points (optional)

- **insertLocalImage** - Upload a local image file to Drive and insert it into a document
  - `documentId`: Document ID
  - `localImagePath`: Absolute path to the local image file
  - `index`: Position to insert at (1-based)
  - `width`, `height`: Image dimensions in points (optional)
  - `uploadToSameFolder`: Upload to same folder as document (optional, default: true)

#### Comments
- **listComments** - List all comments in a Google Document with position context, character offsets, and full reply chains
  - `documentId`: Document ID
  - `includeDeleted`: Include deleted comments (optional, default: false)
  - `pageSize`: Max comments to return, 1-100 (optional, default: 100)
  - `pageToken`: Token for next page of results (optional)
  - Returns surrounding context and Docs API character offsets for each comment using a two-tiered approach (Docs API text matching, DOCX export fallback for ambiguous matches)

- **getComment** - Get a specific comment with its full thread of replies
  - `documentId`: Document ID
  - `commentId`: Comment ID

- **addComment** - Add a comment quoting a text range
  - `documentId`: Document ID
  - `startIndex`: Start index (1-based; or use `textToFind`)
  - `endIndex`: End index (exclusive; or use `textToFind`)
  - `textToFind`: Quote this exact text instead of using indices (case-sensitive)
  - `matchInstance`: Which occurrence of `textToFind` to quote (1-based, default 1)
  - `tabId`: Tab ID for multi-tab documents (defaults to the document body)
  - `commentText`: The comment content
  - Provide either `startIndex`+`endIndex` or `textToFind`
  - The comment appears in the document's comment panel and records the quoted passage. Google Docs does not support margin-anchored comments through any public API: the Docs API exposes no comment surface, and the Drive `anchor` field is [ignored by the Workspace editors](https://developers.google.com/workspace/drive/api/guides/manage-comments), which "treat these comments as un-anchored comments". Comments created here are therefore unanchored by design

- **replyToComment** - Add a reply to an existing comment
  - `documentId`: Document ID
  - `commentId`: Comment ID to reply to
  - `replyText`: The reply content
  - `resolve`: Set to `true` to resolve the comment thread after replying (optional, default: false)

- **deleteComment** - Delete a comment from the document
  - `documentId`: Document ID
  - `commentId`: Comment ID to delete

### Google Sheets

#### Create and Update
- **createGoogleSheet** - Create a Google Sheet
  - `name`: Spreadsheet name
  - `data`: 2D array of cell values
  - `parentFolderId`: Parent folder ID (optional)
  - `valueInputOption`: `RAW` (default, safe) or `USER_ENTERED` (evaluates formulas) (optional)

- **updateGoogleSheet** - Update a Google Sheet
  - `spreadsheetId`: Spreadsheet ID
  - `range`: Range to update (e.g., 'Sheet1!A1:C10')
  - `data`: 2D array of new values
  - `valueInputOption`: `RAW` (default, safe) or `USER_ENTERED` (evaluates formulas) (optional)

- **batchUpdateGoogleSheetValues** - Write many ranges in one `values.batchUpdate` call, across sheets of the same spreadsheet. Cheaper than repeated `updateGoogleSheet` calls: one round trip and one write-quota unit instead of N.
  - `spreadsheetId`: Spreadsheet ID
  - `updates`: Array of `{range, values}` pairs; overlapping ranges have no documented precedence, so avoid them
  - `valueInputOption`: `RAW` (default, safe) or `USER_ENTERED` (evaluates formulas), applied to every range (optional)

- **updateGoogleSheetIfUnchanged** - Write cell values only if the guarded area is unchanged, returning the overwritten contents so the write can be undone
  - `spreadsheetId`: Spreadsheet ID
  - `updates`: ranges to write, ValueRange-shaped; each must name a bounded rectangle (`'Sheet1!A2:C50'` or a single cell), since an open-ended one (`'Sheet1!A2:C'`, `'A:C'`, `'5:9'`) or a bare sheet name would make `preImage` and `postFingerprint` describe a different area than the one written
  - `guardRanges`: ranges that must be unchanged (optional, defaults to the ranges in `updates`; these may be open-ended)
  - `expectedFingerprint`: fingerprint from a previous `dryRun` call; required unless `dryRun`
  - `valueInputOption`: `USER_ENTERED` (default) or `RAW` (optional)
  - `dryRun`: return the fingerprint and guarded contents without writing (optional)
  - `maxCells` / `maxBytes`: budget for the contents returned on a dryRun or refusal (optional)
  - Optimistic, not atomic: the Sheets API has no compare-and-swap, so a write landing between the read and the write is not caught
  - A cell listed in the returned `hazards` cannot be restored by feeding `preImage` back: `updates` carries strings, so re-writing the string re-interprets it as a formula, a number or a boolean and the cell changes type. Its native `userEnteredValue` is returned so that you can restore that one cell outside this server - through the Sheets API's own `updateCells`, or by hand - so an undo that touches a hazard cell needs that cell handled separately
  - The hazard list is deliberately not exhaustive: date-shaped text (e.g. `'2024-01-01'`) and other apostrophe-forced text outside the detected cases (formula-, number-, and boolean-looking) are not flagged

- **getGoogleSheetContent** - Get spreadsheet content with cell information
  - `spreadsheetId`: Spreadsheet ID
  - `range`: Range to get (e.g., 'Sheet1!A1:C10')
  - `valueRenderOption`: `FORMATTED_VALUE` (default, as displayed), `UNFORMATTED_VALUE` (raw values), or `FORMULA` (the formula behind each cell) (optional)
  - Cells within a row are separated by a tab, so a value containing a comma stays one cell
  - Under `UNFORMATTED_VALUE` a date or time cell is returned as a serial number, not a date string

- **getGoogleSheetCells** - Read cells as structured data: one absolute A1 address per cell, the formula and its computed result together, several ranges per call
  - `spreadsheetId`: Spreadsheet ID
  - `ranges`: A1 ranges to read, returned separately and in order
  - `fields`: CellData fields (optional, default `userEnteredValue`, `effectiveValue`, `formattedValue`)
  - `sheetMetadata`: `merges`, `frozen`, `dimensionGroups` (whole sheet), `hiddenRows`, `hiddenColumns`, `dimensionSizes` (only inside the requested ranges) (optional)
  - `includeEmpty`: return cells with none of the requested fields (optional, default false)
  - `maxCells` / `maxBytes`: response budget; on overflow returns `truncated` and `nextRanges` (optional)
  - Addresses are absolute to the sheet, not relative to the requested range, and `getGoogleSheetContent` is unchanged

#### Sheet Management
- **getSpreadsheetInfo** - Get detailed information about a spreadsheet including all sheets/tabs
  - `spreadsheetId`: Spreadsheet ID

- **appendSpreadsheetRows** - Append rows to the end of a sheet
  - `spreadsheetId`: Spreadsheet ID
  - `range`: A1 notation range indicating where to append (e.g., 'A1' or 'Sheet1!A1')
  - `values`: 2D array of values to append
  - `valueInputOption`: `RAW` or `USER_ENTERED` (optional, default: USER_ENTERED)

- **addSpreadsheetSheet** - Add a new sheet/tab to an existing spreadsheet
- **addSheet** - Alias for `addSpreadsheetSheet`
  - `spreadsheetId`: Spreadsheet ID
  - `sheetTitle`: Title for the new sheet

- **listSheets** - List tabs/sheets in a spreadsheet
  - `spreadsheetId`: Spreadsheet ID

- **renameSheet** - Rename a sheet/tab by `sheetId`
  - `spreadsheetId`: Spreadsheet ID
  - `sheetId`: Sheet ID
  - `newTitle`: New title

- **deleteSheet** - Delete a sheet/tab by `sheetId`
  - `spreadsheetId`: Spreadsheet ID
  - `sheetId`: Sheet ID

- **addDataValidation** - Add data validation rules to a range
  - `spreadsheetId`: Spreadsheet ID
  - `range`: A1 range (e.g., `Sheet1!A1:A10`)
  - `conditionType`: `ONE_OF_LIST`, `ONE_OF_RANGE`, `NUMBER_GREATER`, `NUMBER_LESS`, or `TEXT_CONTAINS`
  - `values`: Condition values (e.g. list items, threshold). For `ONE_OF_RANGE`: exactly one value, the source range in A1 notation (e.g. `Reference!A2:A50`); a leading `=` is added automatically if omitted
  - `strict`: Reject invalid values (optional, default: `true`)
  - `showCustomUi`: Show dropdown/custom UI (optional, default: `true`)

- **protectRange** - Protect a range in a spreadsheet
  - `spreadsheetId`: Spreadsheet ID
  - `range`: A1 range
  - `description`: Protection description (optional)
  - `warningOnly`: Warn instead of enforce (optional, default: `false`)

- **addNamedRange** - Create a named range
  - `spreadsheetId`: Spreadsheet ID
  - `name`: Named range name
  - `range`: A1 range

- **listGoogleSheets** - List Google Spreadsheets with optional filtering
  - `query`: Search query to filter by name or content (optional)
  - `maxResults`: Maximum spreadsheets to return, 1-100 (optional, default: 20)
  - `orderBy`: Sort order — `modifiedTime desc` (default), `modifiedTime`, `createdTime desc`, `createdTime`, `recency desc`, `recency`, `name`, or `name_natural`. Keys without `desc` sort ascending, so plain `modifiedTime` is oldest-first (optional)

- **setColumnWidth** - Set column widths in pixels
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `startColumn`, `endColumn`: 0-based half-open column range
  - `pixelSize`: Width in pixels, at least 0

- **setRowHeight** - Set row heights in pixels
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `startRow`, `endRow`: 0-based half-open row range
  - `pixelSize`: Height in pixels, at least 0

- **autoResizeColumns** - Resize columns to fit their contents
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `startColumn`, `endColumn`: 0-based half-open column range

- **autoResizeRows** - Resize rows to fit their contents
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `startRow`, `endRow`: 0-based half-open row range

- **hideSheetDimension** - Hide rows or columns
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `dimension`: `COLUMNS` or `ROWS`
  - `startIndex`, `endIndex`: 0-based half-open range

- **showSheetDimension** - Unhide rows or columns
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `dimension`: `COLUMNS` or `ROWS`
  - `startIndex`, `endIndex`: 0-based half-open range

- **addDimensionGroup** - Group rows or columns into a collapsible outline group
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `dimension`: `COLUMNS` or `ROWS`
  - `startIndex`, `endIndex`: 0-based half-open range
  - Nest groups by adding a second group inside the range of the first

- **deleteDimensionGroup** - Remove a row or column group (the rows/columns are kept)
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `dimension`: `COLUMNS` or `ROWS`
  - `startIndex`, `endIndex`: 0-based half-open range
  - The API decrements the depth of every dimension in the range rather than matching a whole group, so a partially overlapping range shrinks a group instead of removing it. Pass the full range reported by `listDimensionGroups` to remove one

- **updateDimensionGroup** - Collapse or expand an existing group
  - `spreadsheetId`, `sheetId`: Spreadsheet and sheet IDs
  - `dimension`: `COLUMNS` or `ROWS`
  - `startIndex`, `endIndex`: 0-based half-open range
  - `depth`: Nesting depth, 1 for an outermost group (optional, default 1)
  - `collapsed`: `true` to collapse, `false` to expand

- **listDimensionGroups** - List row and column groups with their range, depth and collapsed state
  - `spreadsheetId`: Spreadsheet ID
  - `sheetId`: Limit to one sheet (optional; omit to list every sheet)

#### Formatting
- **formatGoogleSheetCells** - Format cell properties
  - `spreadsheetId`: Spreadsheet ID
  - `range`: Range to format (e.g., 'A1:C10')
  - `backgroundColor`: Cell background color (RGB 0-1) (optional)
  - `horizontalAlignment`: LEFT, CENTER, or RIGHT (optional)
  - `verticalAlignment`: TOP, MIDDLE, or BOTTOM (optional)
  - `wrapStrategy`: OVERFLOW_CELL, CLIP, or WRAP (optional)

- **formatGoogleSheetText** - Apply text formatting to cells
  - `spreadsheetId`: Spreadsheet ID
  - `range`: Range to format (e.g., 'A1:C10')
  - `bold`, `italic`, `strikethrough`, `underline`: Text styling (optional)
  - `fontSize`: Font size in points (optional)
  - `fontFamily`: Font name (optional)
  - `foregroundColor`: Text color (RGB 0-1) (optional)

- **formatGoogleSheetNumbers** - Apply number/date formatting
  - `spreadsheetId`: Spreadsheet ID
  - `range`: Range to format (e.g., 'A1:C10')
  - `pattern`: Format pattern (e.g., '#,##0.00', 'yyyy-mm-dd', '$#,##0.00', '0.00%')
  - `type`: NUMBER, CURRENCY, PERCENT, DATE, TIME, DATE_TIME, or SCIENTIFIC (optional)

- **setGoogleSheetBorders** - Configure cell borders
  - `spreadsheetId`: Spreadsheet ID
  - `range`: Range to format (e.g., 'A1:C10')
  - `style`: SOLID, DASHED, DOTTED, or DOUBLE
  - `width`: Border thickness 1-3 (optional)
  - `color`: Border color (RGB 0-1) (optional)
  - `top`, `bottom`, `left`, `right`: Apply to specific borders (optional)
  - `innerHorizontal`, `innerVertical`: Apply to inner borders (optional)

- **mergeGoogleSheetCells** - Merge cells in a range
  - `spreadsheetId`: Spreadsheet ID
  - `range`: Range to merge (e.g., 'A1:C3')
  - `mergeType`: MERGE_ALL, MERGE_COLUMNS, or MERGE_ROWS

- **addGoogleSheetConditionalFormat** - Add conditional formatting rules
  - `spreadsheetId`: Spreadsheet ID
  - `range`: Range to apply formatting (e.g., 'A1:C10')
  - `condition`: Condition configuration
    - `type`: NUMBER_GREATER, NUMBER_LESS, TEXT_CONTAINS, TEXT_STARTS_WITH, TEXT_ENDS_WITH, or CUSTOM_FORMULA
    - `value`: Value to compare or formula
  - `format`: Format to apply when condition is true
    - `backgroundColor`: Cell color (RGB 0-1) (optional)
    - `textFormat`: Text formatting with bold and foregroundColor (optional)

### Google Slides

#### Create and Update
- **createGoogleSlides** - Create a presentation
  - `name`: Presentation name
  - `slides`: Array of slides with title and content
  - `parentFolderId`: Parent folder ID (optional)

- **updateGoogleSlides** - Update an existing presentation
  - `presentationId`: Presentation ID
  - `slides`: Array of slides with title and content (replaces all existing slides)

#### Content and Formatting
- **getGoogleSlidesContent** - Get presentation content with element IDs
  - `presentationId`: Presentation ID
  - `slideIndex`: Specific slide index (optional)

- **formatGoogleSlidesText** - Apply text formatting to slide elements
  - `presentationId`: Presentation ID
  - `objectId`: Element ID
  - `startIndex`/`endIndex`: Text range (optional, 0-based)
  - `bold`, `italic`, `underline`, `strikethrough`: Text styling (optional)
  - `fontSize`: Font size in points (optional)
  - `fontFamily`: Font name (optional)
  - `foregroundColor`: Text color (RGB 0-1) (optional)

- **formatGoogleSlidesParagraph** - Apply paragraph formatting
  - `presentationId`: Presentation ID
  - `objectId`: Element ID
  - `alignment`: START, CENTER, END, or JUSTIFIED (optional)
  - `lineSpacing`: Line spacing multiplier (optional)
  - `bulletStyle`: NONE, DISC, ARROW, SQUARE, DIAMOND, STAR, or NUMBERED (optional)

- **styleGoogleSlidesShape** - Style shapes and elements
  - `presentationId`: Presentation ID
  - `objectId`: Shape ID
  - `backgroundColor`: Fill color (RGBA 0-1) (optional)
  - `outlineColor`: Border color (RGB 0-1) (optional)
  - `outlineWeight`: Border thickness in points (optional)
  - `outlineDashStyle`: SOLID, DOT, DASH, DASH_DOT, LONG_DASH, or LONG_DASH_DOT (optional)

- **setGoogleSlidesBackground** - Set slide background color
  - `presentationId`: Presentation ID
  - `pageObjectIds`: Array of slide IDs
  - `backgroundColor`: Background color (RGBA 0-1)

- **createGoogleSlidesTextBox** - Create formatted text box
  - `presentationId`: Presentation ID
  - `pageObjectId`: Slide ID
  - `text`: Text content
  - `x`, `y`, `width`, `height`: Position/size in EMU (1/360000 cm)
  - `fontSize`, `bold`, `italic`: Text formatting (optional)

- **createGoogleSlidesShape** - Create styled shape
  - `presentationId`: Presentation ID
  - `pageObjectId`: Slide ID
  - `shapeType`: RECTANGLE, ELLIPSE, DIAMOND, TRIANGLE, STAR, ROUND_RECTANGLE, or ARROW
  - `x`, `y`, `width`, `height`: Position/size in EMU
  - `backgroundColor`: Fill color (RGBA 0-1) (optional)

#### Speaker Notes
- **getGoogleSlidesSpeakerNotes** - Get speaker notes from a slide
  - `presentationId`: Presentation ID
  - `slideIndex`: Slide index (0-based)

- **updateGoogleSlidesSpeakerNotes** - Update or set speaker notes for a slide
  - `presentationId`: Presentation ID
  - `slideIndex`: Slide index (0-based)
  - `notes`: The speaker notes content to set

#### Slide Operations and Templating
- **deleteGoogleSlide** - Delete a slide by object ID
  - `presentationId`: Presentation ID
  - `slideObjectId`: Slide object ID

- **duplicateSlide** - Duplicate a slide by object ID
  - `presentationId`: Presentation ID
  - `slideObjectId`: Slide object ID

- **reorderSlides** - Reorder slides by object IDs and insertion index
  - `presentationId`: Presentation ID
  - `slideObjectIds`: Array of slide object IDs to move
  - `insertionIndex`: Target insertion index

- **replaceAllTextInSlides** - Replace text across a presentation
  - `presentationId`: Presentation ID
  - `containsText`: Text to find
  - `replaceText`: Replacement text
  - `matchCase`: Match case (optional, default: `false`)

- **exportSlideThumbnail** - Export a slide thumbnail URL (PNG/JPEG, SMALL/MEDIUM/LARGE)
  - `presentationId`: Presentation ID
  - `slideObjectId`: Slide object ID
  - `mimeType`: `PNG` or `JPEG` (optional, default: `PNG`)
  - `size`: `SMALL`, `MEDIUM`, or `LARGE` (optional, default: `LARGE`)

- **insertSlidesImageFromUrl** - Insert an image from a public URL
  - `presentationId`, `pageObjectId`: Presentation and slide IDs
  - `imageUrl`: Public image URL
  - `x`, `y`, `width`, `height`: Position and size in EMU (optional)

- **setSlideVisibility** - Show or hide (skip) slides in one atomic call
  - `presentationId`: Presentation ID
  - `slideObjectIds`: Slide object IDs to change
  - `skipped`: `true` = hide (skip), `false` = show

- **replaceSlideImage** - Replace an existing image in place, preserving layering, position, size, and crop
  - `presentationId`, `imageObjectId`: Presentation and image element IDs
  - Source (use one): `imageUrl` (public URL) OR `localImagePath` (uploaded to Drive briefly, then cleaned up)
  - `replaceMethod`: `CENTER_INSIDE` (default) or `CENTER_CROP` (optional)

- **setElementZOrder** - Change element stacking order (`BRING_TO_FRONT`, `BRING_FORWARD`, `SEND_BACKWARD`, `SEND_TO_BACK`)
  - `presentationId`: Presentation ID
  - `pageElementObjectIds`: Element object IDs (all on the same page)
  - `operation`: Z-order operation

- **setElementText** - Replace the entire text of one shape/text box by objectId (element-scoped; cannot hit other slides, masters, or layouts)
  - `presentationId`, `objectId`: Presentation and element IDs
  - `text`: New text content (empty string clears the element)

- **insertSlidesLocalImage** - Upload a local image to Drive and insert it into a slide
  - `presentationId`, `pageObjectId`: Presentation and slide IDs
  - `localImagePath`: Absolute path on the server
  - `x`, `y`, `width`, `height`: Position and size in EMU (optional)

- **getSlideElementInfo** - Get element positions, sizes, transforms, and rendered bounds
  - `presentationId`: Presentation ID
  - `slideObjectId`: Limit results to one slide (optional)

- **moveSlideElement** - Move or resize an image, text box, or shape
  - `presentationId`, `objectId`: Presentation and element IDs
  - `x`, `y`, `width`, `height`: New position or size in EMU (optional)

- **deleteSlideElement** - Delete an image, text box, or shape
  - `presentationId`, `objectId`: Presentation and element IDs

### Google Calendar
- **listCalendars** - List all accessible Google Calendars
  - `showHidden`: Include hidden calendars (optional, default: false)

- **getCalendarEvents** - Get events from a calendar with optional filtering
  - `calendarId`: Calendar ID (optional, default: primary)
  - `timeMin`: Start of time range, RFC3339 (optional, e.g., '2024-01-01T00:00:00Z')
  - `timeMax`: End of time range, RFC3339 (optional)
  - `query`: Free text search in events (optional)
  - `maxResults`: Maximum events to return, 1-250 (optional, default: 50)
  - `singleEvents`: Expand recurring events into instances (optional, default: true)
  - `orderBy`: Sort order — `startTime` or `updated` (optional, default: startTime)

- **getCalendarEvent** - Get a single calendar event by ID
  - `eventId`: Event ID
  - `calendarId`: Calendar ID (optional, default: primary)
  - Response includes the event's file `attachments` (title and URL) when present

- **createCalendarEvent** - Create a new calendar event with Google Meet support
  - `summary`: Event title
  - `start`: Start time (`dateTime` for timed events, `date` for all-day, optional `timeZone`)
  - `end`: End time (same format as start)
  - `calendarId`: Calendar ID (optional, default: primary)
  - `description`: Event description (optional)
  - `location`: Event location (optional)
  - `attendees`: Array of email addresses (optional)
  - `sendUpdates`: `all`, `externalOnly`, or `none` (optional, default: none)
  - `conferenceType`: `hangoutsMeet` to add Google Meet link (optional)
  - `recurrence`: Array of RRULE strings for recurring events (optional)
  - `visibility`: `default`, `public`, `private`, or `confidential` (optional)
  - `attachments`: Array of `{ fileUrl, title?, mimeType? }` (optional, max 25; for Drive files use the file's share URL as `fileUrl`)

- **updateCalendarEvent** - Update an existing calendar event
  - `eventId`: Event ID
  - `calendarId`: Calendar ID (optional, default: primary)
  - `summary`, `description`, `location`: Updated fields (optional)
  - `start`, `end`: Updated times (optional)
  - `attendees`: Updated attendee emails, replaces existing (optional)
  - `attachments`: Array of `{ fileUrl, title?, mimeType? }`, replaces existing (optional, max 25); omit to keep current attachments, or pass `[]` to remove all
  - `sendUpdates`: `all`, `externalOnly`, or `none` (optional, default: none)

- **deleteCalendarEvent** - Delete a calendar event
  - `eventId`: Event ID
  - `calendarId`: Calendar ID (optional, default: primary)
  - `sendUpdates`: Send cancellation notifications (optional, default: none)

### Account Management

These admin tools manage the multi-account state and are always available regardless of tool filtering. They do **not** accept the `account` parameter — see [Multi-account support](authentication.md#multi-account-support) for the full model.

- **manage_accounts** - Add, list, remove, or set the default Google account connection (local OAuth mode only)
  - `action`: one of `list`, `add`, `remove`, `set_default`
  - `account_id`: alias for the account (required for `add`, `remove`, `set_default`). Must match `/^[a-z0-9][a-z0-9_-]{0,31}$/` and not be a reserved name. For `set_default`, pass the literal string `"null"` to clear the default.

The diagnostic tools `authGetStatus`, `authListScopes`, and `authTestFileAccess` are described under [Auth diagnostics](#auth-diagnostics-v170).

### Per-tool `account` parameter

Every non-admin tool carries an optional top-level `account` field. Pass the alias of a connected account to route that specific call there; omit it to fall back to the session/global default or the sole eligible account. See [Per-tool account selection](authentication.md#per-tool-account-selection).
