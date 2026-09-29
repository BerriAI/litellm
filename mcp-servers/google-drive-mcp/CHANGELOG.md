# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **sheets:** `getGoogleSheetCells` reads cells as structured data rather than joined text, taking the server from 128 tools to 129. `getGoogleSheetContent` renders a range as one line per row prefixed `Row N`, and three things are unrecoverable from that: the row numbers count from the start of the requested range rather than from the sheet and the columns are not labelled at all, so an agent that has to write back to a cell it just read must reconstruct the address by counting; `valueRenderOption` picks one rendering for the whole call, so checking a formula against the value it evaluates to takes two round trips that then have to be re-aligned by position; and every value arrives as display text, where an error cell is the literal string `#DIV/0!`, indistinguishable from a cell containing that text. Each cell now comes back as its own object carrying an absolute A1 address and, by default, `userEnteredValue` (the formula), `effectiveValue` (what it computes to, including `errorValue`) and `formattedValue` together. `ranges` takes several A1 ranges across different sheets in one `spreadsheets.get`, each returned separately and in order; `fields` selects any of the nine CellData fields the tool exposes (notes, data validation, hyperlinks, text format runs and the two format objects among them); `sheetMetadata` adds per-sheet extras once per sheet — `merges`, `frozen` and `dimensionGroups` for the whole sheet, `hiddenRows`, `hiddenColumns` and `dimensionSizes` for the rows and columns inside the requested ranges — with merges reported at full extent even when they reach outside the window, since clipping them would hide the fact that they do. `maxCells` (default 2000) and `maxBytes` (default 131072, measured in bytes of this tool's own output) bound the whole response rather than each range: on overflow the read stops at a row boundary and returns `truncated: true` plus `nextRanges`, absolute A1 ranges that can be passed straight back in to continue, so a large read pages instead of failing. `getGoogleSheetContent` is unchanged and remains the right tool for a quick human-readable dump of one range ([#210](https://github.com/piotr-agier/google-drive-mcp/pull/210))
- **sheets:** `updateGoogleSheetIfUnchanged` writes cell values only if a guarded area has not changed since it was last read, and returns what was overwritten so the write can be undone, taking the server from 129 tools to 130. Every other Sheets write is unconditional: it overwrites whatever is there and reports counts rather than contents, so an edit that landed between the read and the write is lost silently and unrecoverably. Docs does not have this problem — `ifRevisionId` rides on the Docs `batchUpdate` as `WriteControl.requiredRevisionId` and the API refuses a stale write itself — but Sheets has no equivalent, established by experiment rather than assumed: a `spreadsheets.batchUpdate` carrying a deliberately wrong `requiredRevisionId` applies anyway, and `spreadsheets.get` exposes no revision field to send in the first place. Call once with `dryRun: true` to get a fingerprint of the guarded area (`guardRanges`, default the ranges being written), then again passing it back as `expectedFingerprint`; a changed area is refused with its current contents attached instead of being overwritten. The response carries `preImage` (a `{range, values}` payload in exactly the shape `updates` takes) and `postFingerprint`, so an undo is this same tool called with the returned pre-image as `updates` and the returned `postFingerprint` as the guard — no rollback state is kept anywhere. Both describe the declared rectangle of each written range, so every range in `updates` has to name one: an open-ended range such as `A2:C` is refused before any API call, because a pre-image read of it is trimmed to the data that happened to be there and would under-cover a write that reaches past it. `guardRanges` may stay open-ended - a guard is read the same way on both sides of the comparison. `hazards` lists cells that cannot be restored by feeding `preImage` back, since `updates` carries strings and re-writing the string re-interprets it (formula-, number-, or boolean-looking text kept as text behind a leading apostrophe), each reported with the native `userEnteredValue` so that the one cell can be restored outside this server - through the Sheets API's own `updateCells`, or by hand; the list is deliberately not exhaustive, and date-shaped text is not detected. Optimistic, not atomic, and the tool description says so in those words: with no compare-and-swap in the API the check happens in this server, so a write landing in the gap between the read and the write (well under a second) is not caught — what is caught is the case that actually happens, a change made before that gap. Two API calls, not three: the guard read doubles as the pre-image read, and the post-write fingerprint comes from the write response rather than a second read ([#217](https://github.com/piotr-agier/google-drive-mcp/pull/217))

## [2.11.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.10.0...v2.11.0) (2026-09-19)

Makes **concurrent Docs editing safe and Docs styling legible**, and tells clients which tools only read. Every Docs write tool now accepts an optional `ifRevisionId`, and every Docs read reports the `revisionId` to pass back, so an agent edit fails cleanly instead of silently clobbering a concurrent human one — the identifier had no route through this server before, because the Drive `version` integer and the ids from `getRevisions` are different things that `WriteControl` rejects. Styling questions previously cost a full-document JSON fetch (~160k characters for three pages); `getGoogleDocStyleSummary` and `describeGoogleDocRange` answer them in a few hundred tokens, reporting real document indices that feed straight back into `applyParagraphStyle` and `applyTextStyle`. `styleDocTable` reaches the table borders, shading, padding, and sizing that nothing here could touch, four Slides wrappers cover slide visibility, in-place image replacement, z-order, and per-element text, and `batchUpdateGoogleSheetValues` writes many ranges in one round trip and one unit of the per-minute write quota — taking the server from 120 tools to 128. The 34 read-only tools also carry the MCP `readOnlyHint` annotation now: the spec defaults `destructiveHint` to true, so `search` and `readGoogleDoc` were indistinguishable from `deleteItem` to a conforming client. On the fix side, `findAndReplaceInDoc` was made trustworthy — exact counting before the write, a diagnosed zero-match, an `expectedCount` guard, real paragraph breaks from an embedded newline, and a lock on the batch it compiles — and `getGoogleSheetContent` separates a row's cells with a tab rather than a comma and a space that nothing could tell apart from a comma inside a cell. Minor rather than major: the contract is additive — eight new tools and new optional parameters that default to the previous behaviour, with nothing removed or renamed. Two changes are observable without opting in: the Docs reads lead or trail with a `revisionId:` line, and `getGoogleSheetContent` rows use the new separator. Both are rendered text rather than a declared contract, which is how 2.10.0 treated the paragraph-meta lines it added, and the old comma-space separator already returned wrong answers for any row holding a prose cell.

### Added

- **protocol:** the 34 read-only tools now carry the MCP `readOnlyHint` annotation. The spec defaults `destructiveHint` to true, so a server that annotates nothing has every tool treated as destructive by a conforming client — `search`, `readGoogleDoc` and `getDocumentInfo` were indistinguishable from `deleteItem`, which is what drives confirmation prompts in clients that honour the hints. The flag is projected from the read/write/admin classification the account resolver already uses, corrected at two edges where that classification answers a different question: `downloadFile` needs only a read scope but writes (and with `overwrite`, replaces) a file on the host, so it is not advertised as read-only, and the three `auth*` diagnostics are admin-dispatched but only ever read, so they are. `destructiveHint` is deliberately still unset: marking an additive write non-destructive takes a per-tool judgement across 92 tools, and a wrong call there would suppress a prompt that should have fired, so the spec's safe default stands until that pass happens. Annotations are hints — the spec requires clients to treat them as untrusted unless the server is trusted — so this improves the experience in cooperating clients rather than enforcing anything ([#216](https://github.com/piotr-agier/google-drive-mcp/pull/216))
- **docs:** every Google Docs write tool accepts an optional `ifRevisionId`, an optimistic lock that fails the write cleanly if the document changed since the caller read it, so an agent edit cannot silently clobber a concurrent human one. The value is the Docs API `revisionId`, which nothing in this server previously reported — the Drive `version` integer from `getDocumentInfo` and the revision ids from `getRevisions` are different identifiers that `WriteControl` rejects — so the reads now surface it: `readGoogleDoc` leads its `text`/`markdown` output with a `revisionId:` line (counted against `maxLength`, so a truncated read is still lockable), `getGoogleDocContent` trails with one, `getGoogleDocContentPaginated` carries a `revisionId` field, and `getDocumentInfo` reports the Docs `revisionId` alongside the Drive version it already showed, each labelled. A caller without edit access gets `unavailable (no edit access)` rather than a silently missing value, since Docs only populates the field for editors. A revisionId is valid for 24 hours and cannot be shared across users, which matters in multi-account mode: read and write under the same account. Omitting the parameter leaves every tool behaving exactly as before ([#187](https://github.com/piotr-agier/google-drive-mcp/pull/187), [#208](https://github.com/piotr-agier/google-drive-mcp/pull/208))
- **docs:** two read-side style tools, so styling questions stop costing a full-document JSON fetch (~160k characters for a three-page document) — the reads that previously had no answer short of applying a style just to read its range back. `getGoogleDocStyleSummary` returns a whole-document inventory in a few hundred tokens: fonts with their size ladders, named-style counts, text colors, bordered and shaded paragraph locations, a table inventory with real index ranges, and the heading outline. `describeGoogleDocRange` probes one spot, reporting the paragraph styles (named style, alignment, borders, shading, indents, spacing, bullets) and text runs (font, size, bold/italic/underline, colors, links) overlapping a range, each with its real document index so the output feeds straight into `applyParagraphStyle` or `applyTextStyle`; target it by `startIndex` (optionally `endIndex`) or by `textToFind`, and mixing the two modes is refused rather than silently resolved, as `insertText` and `deleteRange` already do. Both respect the Docs index model rather than merging index spaces: headers, footers, and footnotes each restart at 0 and are excluded, and indices restart in every tab, so `describeGoogleDocRange` describes one tab (the first unless given a `tabId`) instead of reporting several paragraphs with identical ranges and no way to tell which one a write should target. The summary keeps its counts and font inventory whole-document but prefixes each location with the tab title when a document has more than one tab, and truncates every long list the same way, with an explicit `… (+N more)` — a silently clipped list is indistinguishable from a short one. Only a positive-width border counts toward the bordered total, since Docs attaches empty border objects to most paragraphs, which made a nine-rule document report 97. Both report the `revisionId` from the same fetch the indices came from, so the read that locates an edit also supplies the `ifRevisionId` that guards it ([#184](https://github.com/piotr-agier/google-drive-mcp/pull/184), [#211](https://github.com/piotr-agier/google-drive-mcp/pull/211))
- **docs:** `styleDocTable` styles a table in one atomic `batchUpdate`: cell borders, background, padding, and content alignment — over the whole table, or over a cell range given `rowIndex`+`columnIndex` with optional `rowSpan`/`columnSpan` — plus fixed column widths (`columnWidth`, scoped by `columnIndices`, minimum 5pt) and minimum row heights (`minRowHeight`, scoped by `rowIndices`). A table arrives with a default 1pt grid that nothing in the server could touch, and `editTableCell` reaches only one cell's text and character styling. Clearing a border is expressed as an explicit width-0 border rather than a field reset, matching the Docs UI's "0 pt" unbox, so `removeBorders: ["all"]` unboxes a table in a single call. Cell targeting and the column/row parameters are refused together, since the latter always apply at table scope and would otherwise widen silently past the cell the caller named. Take `tableStartIndex` from the table's span line in `readGoogleDoc` or `getGoogleDocContent`, which prints `tableStartIndex=N`. Accepts `ifRevisionId` ([#186](https://github.com/piotr-agier/google-drive-mcp/pull/186), [#209](https://github.com/piotr-agier/google-drive-mcp/pull/209))
- **slides:** four wrappers over request families the server did not expose. `setSlideVisibility` flips `isSkipped` on any number of slides in one atomic batch, where hiding or showing previously meant one manual action per slide. `replaceSlideImage` swaps an image in place via `replaceImage`, preserving z-order, position, size, and crop — delete-and-reinsert lands the new image on top of the stack, covering any text or panel laid over it — and accepts a public URL or a local file, uploaded briefly to Drive and cleaned up afterwards. `setElementZOrder` exposes `updatePageElementsZOrder` for the same stacking problem on elements already in place. `setElementText` replaces one element's entire text by `objectId` in a single batch, scoped to that element: unlike `replaceAllTextInSlides` it cannot reach other slides, layouts, or masters, and does not depend on matching the existing text. It skips `deleteText` on a shape that holds no text, which the API rejects, and counts an `autoText` element (a slide-number or date placeholder) as text so the replacement lands in place of it rather than in front of it ([#186](https://github.com/piotr-agier/google-drive-mcp/pull/186), [#209](https://github.com/piotr-agier/google-drive-mcp/pull/209))
- **sheets:** `batchUpdateGoogleSheetValues` writes many ranges of one spreadsheet in a single `spreadsheets.values.batchUpdate` call. Ranges may sit on different sheets, `valueInputOption` (`RAW` by default, `USER_ENTERED` to evaluate formulas) applies to the whole batch, and the reply reports cell and range counts rather than echoing the values back. One batch costs one round trip and one unit of the per-minute write quota where N `updateGoogleSheet` calls cost N of each ([#203](https://github.com/piotr-agier/google-drive-mcp/pull/203), [#206](https://github.com/piotr-agier/google-drive-mcp/pull/206))

### Fixed

- **docs:** `findAndReplaceInDoc` now locks the batch it compiles for a multi-line `replaceText`. That path sent its delete+insert pairs without `writeControl`, so an `ifRevisionId` was accepted and never applied, and a concurrent edit between the tool's read and its write shifted every index the batch targeted. The batch now carries the caller's `ifRevisionId`, or otherwise the `revisionId` of the read the matches came from, so a changed document fails the write instead of taking the replacement at the wrong offsets. That failure is reported as a message naming the cause and telling the caller to re-read, rather than the API's raw 400. The mapping keys on whether the write carried a lock, not on the wording of Google's error: the 400 for a `requiredRevisionId` mismatch is documented but its message text is not, so a condition matching that text would have stopped firing silently if Google reworded it. The API's own message is kept alongside, so a 400 from any other cause stays diagnosable ([#212](https://github.com/piotr-agier/google-drive-mcp/pull/212), [#219](https://github.com/piotr-agier/google-drive-mcp/pull/219))
- **docs:** `insertText` and `deleteRange` now declare `ifRevisionId` in their input schemas. Both handlers already applied it and `docs/tools.md` already listed it, but a client building calls from `listTools` could not see the parameter, which left the two index-based writes most exposed to a concurrent edit without a lock a model could find ([#213](https://github.com/piotr-agier/google-drive-mcp/pull/213))
- **docs:** `findAndReplaceInDoc` now reports what it actually did, and a `replaceText` containing `\n` produces real paragraph breaks. Three problems, one tool: the reported count came from `replaceAllText`'s reply with no way to check it beforehand, a zero-match replace was a silent no-op that reads exactly like the text not being there, and an embedded newline went to `replaceAllText`, whose newline handling flattens the surrounding paragraphs and drops characters. Occurrences are now counted exactly before the write, across the same surface `replaceAllText` targets — body, tables (recursive), headers, footers, footnotes, and every tab or a single `tabId`. A zero-match result names the likeliest lookalike cause, with the codepoints spelled out: non-breaking spaces (how Docs fakes letterspacing), curly versus straight quotes, literal `&amp;` entities, and case. The new optional `expectedCount` is a pre-write guard — the call aborts without touching the document when the count disagrees, so a substring collision becomes a refused call rather than a replacement to undo, and an overshoot says so explicitly. A `replaceText` containing `\n` is compiled to exact `deleteContentRange` + `insertText` pairs applied descending in one atomic batch, so `\n` renders as real paragraph breaks; `findText` must stay single-line, since a match cannot span a paragraph break, and a case-insensitive match is mapped back through the fold before it becomes a document index (lowercasing is not length-preserving — U+0130 expands to two UTF-16 units, which would otherwise shift every index after it). `expectedCount` costs one extra document read per call, and a zero-match result costs one more to diagnose ([#181](https://github.com/piotr-agier/google-drive-mcp/pull/181), [#207](https://github.com/piotr-agier/google-drive-mcp/pull/207))
- **docs:** `updateGoogleDoc` no longer replaces content as two separate calls. The delete and the insert now go out as one atomic `batchUpdate`, matching the `tabId` path, so a failed insert can no longer leave the document wiped. This also closes a gap in the new `ifRevisionId` guard: the lock rode on the delete, and an empty document skips the delete entirely, so the write went through unguarded on exactly the documents where the check was cheapest to get wrong. `insertText` and `deleteRange` now refuse `ifRevisionId` on plain text files instead of accepting it and ignoring it, matching how they already refuse `tabId` ([#187](https://github.com/piotr-agier/google-drive-mcp/pull/187), [#208](https://github.com/piotr-agier/google-drive-mcp/pull/208))
- **sheets:** `getGoogleSheetContent` now separates the cells of a row with a tab rather than a comma and a space, under every `valueRenderOption`. The old separator could not be told apart from a comma and a space inside a cell, so a three-cell row such as `North, $1,234.56, Up, but slowing` offered four plausible split points and nothing said which three were real. Reading formulas made this routine rather than occasional, because a formula written with spaces after its commas (`=IF(A1>0, "yes", "no")`) broke the same way; a currency value was never affected, since its thousands comma carries no following space. The `Row N: ` prefix, the range preamble, and the empty-range text are unchanged. Minor rather than major: the rendered text is not a declared contract, and the only exposure is a client splitting on comma-space, which already returned wrong answers for any row holding a prose cell. `deleteDimensionGroup` no longer claims to delete the group it names: the API decrements the group depth of every dimension in the range, so a range that only partially overlaps a group shrinks it instead of removing it (a depth-1 group over B:E and a depth-2 group over C:D, with D:E deleted, leaves depth-1 over B:D and depth-2 over C:C), and the description now says so and points at `listDimensionGroups` for the full range. The `UNFORMATTED_VALUE` description warns that a date or time cell comes back as a spreadsheet serial number, because no `dateTimeRenderOption` is sent. `hideSheetDimension`, `showSheetDimension`, `addDimensionGroup`, and `deleteDimensionGroup` now share one dimension-range schema so their validation cannot drift, the four group tools report `on sheet N` to match the older dimension tools, and an unknown sheet id reads `Sheet with id N not found` ([#205](https://github.com/piotr-agier/google-drive-mcp/pull/205))

## [2.10.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.9.0...v2.10.0) (2026-09-15)

Makes **formulas readable and sheet outlines manageable**, and lets the Docs indexed reads report paragraph-level styling. `getGoogleSheetContent` could only ever return computed values — writing a formula worked, reading it back did not — so it gains a `valueRenderOption` that selects raw values or the formula behind each cell. Row and column outline groups, previously uncovered, gain four tools, taking the server from 116 to 120. On the Docs side, `includeFormatting` reported text-run formatting per span but said nothing about the paragraph itself, so a caller restyling a document had to fall back to the full document JSON; each paragraph carrying non-default styling now gets one `¶` line over its real index range, which can be fed straight to `applyParagraphStyle`. Minor rather than patch: the additions are a new optional parameter that defaults to the previous behavior and four new tools, with no tool, parameter, or configuration option removed or renamed. The only change an existing caller sees is the extra `¶` lines, and only in a read that already opted in to `includeFormatting`.

### Added

- **docs:** `getGoogleDocContent` and `getGoogleDocContentPaginated` with `includeFormatting` now annotate paragraph-level styling, which the indexed reads never exposed. Text-run formatting (font, size, bold/italic/underline/strikethrough, colors, baseline) was already reported per span, but nothing said whether a paragraph was a heading, centered, ruled, or shaded — so a caller restyling a document had to fetch the full document JSON to find out. Each paragraph carrying non-default paragraph styling now gets one `¶` line over its real index range listing the named style, alignment, visible borders (width, color, dash style), and shading. Empty border objects, which Docs attaches to most paragraphs, are skipped — only a positive width renders — and a shading color that resolves to nothing is omitted rather than printed as `shading(null)`. The annotation never enters a table cell's rendering, where it would otherwise be joined into the pipe row as `| ¶ center Header A |` ([#184](https://github.com/piotr-agier/google-drive-mcp/pull/184), [#199](https://github.com/piotr-agier/google-drive-mcp/pull/199))
- **sheets:** `getGoogleSheetContent` accepts `valueRenderOption` — `FORMATTED_VALUE` (default, unchanged), `UNFORMATTED_VALUE`, or `FORMULA` — so a formula can be read back and not merely written. Writing one already worked through `valueInputOption: USER_ENTERED`, but the read called `values.get` with no render option and so always returned the formatted result: a client could see what a formula computed, never the formula that produced it. Row and column outline groups, which had no coverage at all, get `addDimensionGroup`, `deleteDimensionGroup`, `updateDimensionGroup` (collapse or expand by `depth`) and `listDimensionGroups`, taking the server from 116 tools to 120. All four use the same 0-based half-open `startIndex`/`endIndex` convention as the existing dimension tools, and `listDimensionGroups` reports each group's range, depth, and collapsed state in that same form, so its output feeds straight back into the other three. Nest a group by adding a second one inside the range of the first ([#202](https://github.com/piotr-agier/google-drive-mcp/pull/202))

## [2.9.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.8.0...v2.9.0) (2026-09-07)

Completes the Docs **paragraph-styling and text-targeting** surface. `applyParagraphStyle` now reaches the rest of the Docs `ParagraphStyle` — per-edge borders, paragraph shading, first-line indent, and the pagination controls — and `insertText`/`deleteRange` accept `textToFind` + `matchInstance` in place of numeric indices, using the same split-run-aware locator as `applyTextStyle` and `createParagraphBullets`, so targeting behaves the same way across the toolset. Also fixes four places where the Docs and HTTP surfaces misreported their own state: the indexed reads printed fabricated index ranges for table rows, `addComment` sent an anchor Docs cannot resolve (rendering the thread under "Original content was deleted"), an expired HTTP session id was answered with 400 instead of the 404 that tells a client to re-initialize, and a malformed hex color was silently coerced rather than rejected. Minor rather than patch: the release adds new optional parameters to existing Docs tools, and no tool, parameter, or configuration option was removed or renamed.

### Added

- **docs:** `applyParagraphStyle` (and its `formatGoogleDocParagraph` alias, which now shares one input schema so the two cannot drift) exposes the rest of the Docs `ParagraphStyle`: per-edge paragraph borders (`borderTop`/`borderBottom`/`borderLeft`/`borderRight`/`borderBetween`, each `{ color?, width?, padding?, dashStyle? }` with omitted subfields defaulting to a black, 1pt, 1pt-padding, solid line because the API refuses a partially specified border), `removeBorders` to clear edges at the paragraph level so they fall back to the named style, `shading`/`removeShading` for paragraph background color, `indentFirstLine`, and the `keepLinesTogether`/`avoidWidowAndOrphan`/`pageBreakBefore` pagination controls. The update styles every paragraph overlapping the range, so one call over the whole body with `removeBorders: ["all"]` strips every paragraph border from a document ([#185](https://github.com/piotr-agier/google-drive-mcp/pull/185))
- **docs:** `insertText` and `deleteRange` accept `textToFind` + `matchInstance` as an alternative to numeric indices on Google Docs, using the same split-run-aware locator as `applyTextStyle`, `applyParagraphStyle`, and `createParagraphBullets` so targeting behaves identically across the toolset. `insertText` adds `position` (`before` | `after`, default `after`) to say which side of the match receives the text; `deleteRange` deletes the matched range. Matching is exact and case-sensitive. Exactly one targeting mode is accepted per call, and a lone `startIndex` or `endIndex` next to `textToFind` is refused rather than silently ignored. Google Docs refuses to delete a segment's final paragraph break or to insert at the segment end, so a match that runs to the end of the document is trimmed to keep that break on delete and an `after` insert lands just before it; a match that is only that break is reported as an error instead of a Docs API failure. Text files keep 0-based offsets and reject `textToFind` ([#183](https://github.com/piotr-agier/google-drive-mcp/pull/183), [#198](https://github.com/piotr-agier/google-drive-mcp/pull/198))

### Fixed

- **docs:** hex colors are now validated strictly wherever the Docs tools accept one (`foregroundColor`, `backgroundColor`, border `color`, `shading`). Previously a value such as `#12345G` passed the length check and was silently parsed as `#012345`; it is now rejected as an invalid hex color
- **docs:** the indexed reads no longer print made-up index ranges for table rows, and text inside a table is still addressable. `getGoogleDocContent` and `getGoogleDocContentPaginated` derived a range for each rendered table line from the length of the markdown they had just produced, starting at the table's own real start index and running forward past its end into whatever followed — so the numbers printed beside a table's rows pointed at the content after it, sometimes a later table, and an edit made against them landed in the wrong place. A table now emits one real `[start-end]` span for the whole element, followed by the pipe rendering and a `cells: r0c0 [a-b], r0c1 [c-d]` map of each cell's true index range taken from the API, so `applyTextStyle` and `formatGoogleDocText` can still target part of a cell (`editTableCell` styles only a whole cell). The accompanying hint names `tabId` as well as `tableStartIndex`, and the tab header prints that id (`=== Tab: Name (tabId=…) ===`), because index spaces restart per tab and `editTableCell` given no `tabId` searches only the first one. Because table rows carry no index of their own, `getGoogleDocContentPaginated` now snaps a page boundary to the start of the whole table block rather than splitting it; a table longer than `limit` still takes a hard cut so pagination advances, but never between the header line and its first row ([#179](https://github.com/piotr-agier/google-drive-mcp/pull/179))
- **docs:** `addComment` no longer sends an anchor Google Docs cannot resolve. The tool built a Drive anchor (`{'r': documentId, ...}`) by hand, but Workspace editor apps treat API-supplied anchors as un-anchored comments, so no anchor value can produce a margin anchor — and sending an unresolvable one is worse than sending none, because the editor then renders the thread under "Original content was deleted", making the comment read as if the passage it refers to had been removed. The anchor is dropped. `quotedFileContent` is what ties a comment to its passage and what `listComments` matches on to report character positions, so it is now extracted with `buildFlatTextFromDoc`, which walks nested tables and resolves per-tab index spaces — the previous scan looked only at top-level body paragraphs, so a range inside a table quoted an empty string and left the comment with no recoverable position — and it is declared `text/plain` rather than the `text/html` it never was. `addComment` also accepts `textToFind` + `matchInstance` and `tabId` for targeting, and a range covering no text is rejected instead of creating a context-free comment ([#196](https://github.com/piotr-agier/google-drive-mcp/pull/196))
- **transport:** an HTTP client whose session has expired can now recover on its own instead of failing every call until the user reconnects by hand. The Streamable HTTP transport evicts a session after 30 idle minutes, and the spec says a request that presents a session id the server no longer knows must be answered with HTTP 404 so the client starts a new session with `initialize`. The server answered 400 instead, which compliant clients treat as a plain request error, so a connector left idle between chat turns (a claude.ai custom connector, for example) came back to a dead session id and stayed stuck on it. `POST`, `GET`, and `DELETE /mcp` now return 404 with the SDK's own JSON-RPC error (`-32001`, `Session not found`) for an unknown, evicted, or closed session id; a request that omits the `Mcp-Session-Id` header where one is required still gets 400. An `initialize` that still carries a stale session id is accepted and opens a new session, so a client recovering from the 404 does not have to scrub the header first. In team mode a session id belonging to another user is answered with the same 404 as an unknown one, so the hijack guard still does not reveal whether the id exists

## [2.8.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.7.1...v2.8.0) (2026-09-03)

Stops a stalled OAuth token refresh from **hanging every call on the server**. The proactive refresh that runs five minutes before an access token expires awaited Google's token endpoint with no timeout, and because concurrent refreshes of the same account are deduplicated, one stalled request silently froze every tool call behind it. Refreshes now run under the shared retry helper with a per-attempt deadline that actually aborts the underlying token POST, and a refresh that stalls through both attempts fails the call with an explicit error instead of waiting. Minor rather than patch because the bound is configurable: the new `--token-refresh-timeout` / `GOOGLE_DRIVE_MCP_TOKEN_REFRESH_TIMEOUT` sets the per-attempt limit (default `15000`, `0` disables). No tools or tool parameters were added, removed, or renamed; deployments that leave the new option unset get the 15-second default in place of the previous unbounded wait.

### Fixed

- **auth:** an OAuth token refresh that stalls no longer hangs every call on the server. The proactive refresh that runs five minutes before an access token expires awaited Google's token endpoint with no timeout, and because concurrent refreshes of the same account are deduplicated, every tool call waited on the same never-settling request — four-minute hangs that started on an ordinary unrelated call, raised no error, and cleared on restart only because the restart moved the token clock away from the boundary. Refreshes now run under the shared retry helper with a per-attempt deadline and at most one retry, and the deadline actually aborts the underlying token POST (through a gaxios request interceptor scoped to the token endpoint) rather than merely abandoning the wait — without that, the library's own refresh deduplication would pin every retry, and the next API call's implicit refresh, to the stalled request. A refresh that stalls through both attempts fails the current call with an explicit error naming the account and the limit; requests the library issues on its own get the same bound; `invalid_grant` still surfaces on the first attempt with the existing reconnect message; other transient refresh failures behave as before. Team members' per-user refreshes and an external-token refresh get the same treatment. New `--token-refresh-timeout=<ms>` / `GOOGLE_DRIVE_MCP_TOKEN_REFRESH_TIMEOUT` (default `15000`, `0` disables) sets the per-attempt limit; `--retry-base-delay` governs the backoff and `--retry-max=0` disables the retry ([#169](https://github.com/piotr-agier/google-drive-mcp/issues/169))

## [2.7.1](https://github.com/piotr-agier/google-drive-mcp/compare/v2.7.0...v2.7.1) (2026-09-03)

Makes the server **work on Node 24** and stop reporting permission writes it never verified. OAuth token refresh failed outright on Node 24 (`Premature close`) because the Google client stack still resolved `gaxios` 6 / `node-fetch` 2; the clients are upgraded to `gaxios` 7 and CI now runs Node 22 and 24. Separately, Drive's upsert semantics can apply a *different* role than the one requested when the principal already holds a permission, and `addPermission`, `updatePermission`, and `shareFile` all echoed the request back as success — they now verify the role Drive actually applied. Fixes only: no tool, parameter, or configuration changes.

### Fixed

- **deps:** upgrade the Google API clients to `gaxios` 7 (`google-auth-library@^10.2.0`, `googleapis@^173.0.0`) so OAuth token refresh works on Node 24. `gaxios` 6 uses `node-fetch@2`, and that combination fails against Google's token endpoint under Node 24 with `Premature close`. Drops the unused `@google-cloud/local-auth` (which pinned `google-auth-library@^9`) and dedupes the tree to a single `google-auth-library` copy through an npm override; `node-fetch@2` is gone. Also declares `engines.node >= 18`, normalizes media downloads through a stream helper that accepts either a Node `Readable` or a web `ReadableStream`, walks the error cause chain in the retry policy (and accepts undici's `UND_ERR_CONNECT_TIMEOUT` / `UND_ERR_SOCKET`), and prefers `gaxios` 7's numeric `error.status` when mapping Docs 404/403. CI now runs Node 22 and 24, and `test/gaxios-contract.test.ts` pins the response and error shapes the code duck-types against the real library ([#166](https://github.com/piotr-agier/google-drive-mcp/issues/166), [#176](https://github.com/piotr-agier/google-drive-mcp/pull/176))
- **docs:** `getGoogleDocImage` now reports the real image MIME type instead of always falling back to `application/octet-stream`. `gaxios` 7 exposes `response.headers` as a `Headers` instance, and the bracket access used to read `content-type` silently yielded `undefined` on it — mislabelling every image, including in the MCP `type: "image"` block clients render. Header reads accept both the `Headers` and plain-object shapes, case-insensitively
- **auth:** fail fast on a malformed service account key file. `google-auth-library` v10 no longer rejects unparseable or field-less credentials files — `getClient()` resolves with a credential-less JWT client — so the server logged "Service account authentication successful" while holding no credentials and every later call failed with a misleading "Method doesn't allow unregistered callers". The key file is now validated before the client is constructed, naming the offending file and missing fields without echoing key material. Non-service-account ADC types (`authorized_user`, `external_account`, …) are still accepted, and a missing file still surfaces as `ENOENT`
- **drive:** `addPermission`, `updatePermission`, and `shareFile` now verify the role Drive actually applied instead of echoing the requested one. Drive's upsert semantics can apply a different role than requested when the principal already holds a permission on the file — a `reader` grant could come back as `writer` while the response read like success. All three tools now compare Drive's response against the request, apply one corrective `permissions.update`, and report a surviving mismatch as an error rather than success; a response carrying no role at all is treated as a mismatch everywhere. A correction is now described as the existing permission's role being *changed to* the requested one, and the `addPermission`/`shareFile` descriptions state that an existing principal's role is set to the requested one even when that is a downgrade ([#178](https://github.com/piotr-agier/google-drive-mcp/pull/178), [#191](https://github.com/piotr-agier/google-drive-mcp/pull/191))

## [2.7.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.6.0...v2.7.0) (2026-09-02)

Makes team-mode logs **attributable to a person**. Every team-mode log line — dispatch, session lifecycle, sign-in callback, and per-user tool logs — now carries the acting member's Google `sub` and email, so an audit trail can be read without joining opaque identifiers against the team store. Also hardens the per-user logger's error payloads against `JSON.stringify` mangling an `Error` into `{}` and against serializing a gaxios error's raw request config. Additive — no tool, parameter, or configuration changes, and single-user deployments log exactly as before.

### Features

- **team:** log lines now name the acting user. Team-mode dispatch, session lifecycle (created / idle timeout / closed), sign-in callback, and per-user tool logs carry the member's Google `sub` and email, so an audit can tell who did what without joining an opaque `sub` against the team store. Error payloads passed to the per-user logger are rendered through the existing redaction helper instead of `JSON.stringify` (which turns an `Error` into `{}` and would serialize a gaxios error's raw request config). Single-user logs are unchanged ([#177](https://github.com/piotr-agier/google-drive-mcp/pull/177))

## [2.6.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.5.0...v2.6.0) (2026-08-21)

Makes Drive listings **ordered and self-describing**. `search` gains an `orderBy` and sorts most-recently-modified first by default instead of returning Drive's arbitrary order, `listGoogleDocs`/`listGoogleSheets` stop defaulting to *oldest* first, and all three name their effective ordering in the response — so a caller is told the list is sorted rather than left to work it out by comparing timestamps by eye. Also adds superscript/subscript (`baselineOffset`) to the Docs text-styling tools, on both the write and the formatted-read path. Additive — a new optional parameter on existing tools plus corrected default orderings, with no removed or renamed tools/parameters. Also carries the MCP Registry publication work prepared as 2.5.1 but never released, which reaches npm for the first time here.

### Features

- **drive:** `search` now sorts results and accepts an `orderBy` parameter. Results default to `modifiedTime desc` (most recently modified first) instead of Drive's unspecified default order, and the response header names the effective ordering (`Found N files (ordered by modifiedTime desc):`). Previously an unsorted, unlabeled result set left the caller to pick "the most recent" by comparing timestamps across up to 100 rows by eye — behind an LLM client that silently produced wrong answers, citing months-old files as the newest. `orderBy` accepts `modifiedTime desc`, `modifiedTime`, `createdTime desc`, `createdTime`, `recency desc`, `recency`, `name`, and `name_natural`; keys without `desc` sort ascending. When more results are available the response repeats the effective `orderBy` alongside the `pageToken`, so a follow-up page keeps the ordering instead of falling back to the default ([#167](https://github.com/piotr-agier/google-drive-mcp/issues/167))
- **docs:** add `baselineOffset` (`SUPERSCRIPT`/`SUBSCRIPT`/`NONE`) to `applyTextStyle` for superscript and subscript text, and surface it on the formatted read path — `getGoogleDocContent`/`getGoogleDocContentPaginated` with `includeFormatting: true` now emit a `baseline=superscript`/`baseline=subscript` marker so styled runs are distinguishable from plain text on readback ([#158](https://github.com/piotr-agier/google-drive-mcp/pull/158))

### Fixed

- **docs/sheets:** `listGoogleDocs` and `listGoogleSheets` no longer default to *oldest* first. Both passed `orderBy: "modifiedTime"`, which Drive sorts ascending, while advertising only "Sort order for results" — so the default listing surfaced the least recently touched files. Both now default to `modifiedTime desc`, accept the same `orderBy` values as `search` (a superset of the previous three; existing values keep their meaning), and name the ordering in the response header
- **auth:** re-authenticating an existing account with `manage_accounts add <alias>` now takes effect immediately instead of requiring a server restart. The refreshed grant was persisted correctly, but the OAuth client cached for that alias — and the Drive/Calendar services built on it — were left in place, so every subsequent call kept using the superseded (often revoked) grant and failed with "authorization was revoked or has expired". A completed re-consent now evicts those cached clients, the way removing an account already did ([#168](https://github.com/piotr-agier/google-drive-mcp/issues/168))
- **docs:** `formatGoogleDocText` now advertises `baselineOffset` in its input schema. The alias shares a handler and validation schema with `applyTextStyle`, so the parameter already worked at runtime, but it was missing from the advertised tool definition — clients that build arguments from (or validate against) the published schema could not reach superscript/subscript through the alias

### Distribution

- Publish verified npm/stdio metadata to the official MCP Registry under `io.github.piotr-agier/google-drive-mcp`, with live schema validation and GitHub OIDC publication integrated into the release workflow. Prepared as 2.5.1 in July but never released, so it reaches npm for the first time in this release

## 2.5.1 (2026-07-17)

Prepared but never released — there is no `v2.5.1` tag or GitHub release, and npm went
straight from 2.5.0 to 2.6.0. Its changes are listed under 2.6.0.

## [2.5.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.4.0...v2.5.0) (2026-07-15)

Surfaces **embedded inline images** in Google Docs instead of silently dropping them: the read tools now emit a self-describing image token, and a new **`getGoogleDocImage`** tool fetches an image's bytes on demand for OCR/vision workflows. Additive — no removed or renamed tools/parameters; existing single-user deployments are unaffected.

### Features

- **docs:** surface embedded inline images instead of dropping or opaquely placeholdering them. `readGoogleDoc`/`readGoogleDocPaginated` now render each inline image (markdown → `![alt](contentUri "objectId=…")`; text → a single-line `[image: objectId=… contentUri=… sourceUri=… size=WxHpt]` token) instead of silently omitting it, and `getGoogleDocContent`/`getGoogleDocContentPaginated` upgrade the bare `[image]` placeholder to the same self-describing token (objectId, contentUri/sourceUri, size, alt text). A new **`getGoogleDocImage`** tool fetches an inline image's bytes by `(documentId, inlineObjectId)` — it re-fetches the doc to resolve a fresh, non-expired image URL and returns a native MCP image block (or, with `outputFormat: "base64"`, a `{ inlineObjectId, mimeType, byteLength, dataBase64 }` envelope) so downstream OCR/vision/forwarding workflows can reach the content. Keyed by the durable objectId (never a raw contentUri, which expires ~30 min). Floating/anchored images stored as `positionedObjects` remain unrendered (documented limitation) ([#132](https://github.com/piotr-agier/google-drive-mcp/issues/132))

## [2.4.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.3.0...v2.4.0) (2026-07-15)

Adds opt-in **team mode**: an MCP-spec OAuth 2.1 authorization server for multi-user HTTP deployments, so a single running server can be shared by a team (e.g. through claude.ai custom connectors) with each member authenticated individually and every tool call running as the caller. Purely additive — default stdio/HTTP behavior for existing single-user deployments is unchanged.

### Features

- **team:** add opt-in **team mode** for multi-user HTTP deployments (`--team` / `MCP_TEAM_MODE` + `--issuer-url`): the HTTP transport becomes an MCP-spec OAuth 2.1 authorization server (Dynamic Client Registration, PKCE S256, refresh-token rotation with reuse detection, RFC 8414/9728 discovery metadata) doing two-hop OAuth — MCP client ⇄ this server ⇄ Google. Each team member signs in with their own Google account (optionally restricted with `MCP_TEAM_ALLOWED_DOMAINS`, enforced on the Google-asserted `hd` claim); every `/mcp` request is authenticated with an opaque bearer token (SHA-256-hashed at rest) and every tool call runs as the caller. Per-user Google refresh tokens persist in `team-store.json` (mode `0600`, pluggable in-memory/file store). Sessions are bound to the signing-in user; `manage_accounts`, the `account` parameter, and the resources capability are disabled in this mode; a revoked Google grant self-heals by forcing the connector through a fresh OAuth flow. Requires a "Web application" OAuth client and an https issuer URL; designed for claude.ai custom connectors. Behind a reverse proxy, set `MCP_TRUST_PROXY` to the trusted hop count — left unset, per-user rate limiting collapses to one shared bucket and express-rate-limit logs `ERR_ERL_UNEXPECTED_X_FORWARDED_FOR`; the server now warns at startup when it's unset with a non-localhost issuer. Default stdio/HTTP behavior is unchanged ([#109](https://github.com/piotr-agier/google-drive-mcp/issues/109))

### Bug Fixes

- **auth:** error logs in the auth layer no longer leak credential material — gaxios errors were logged whole, embedding the token-refresh POST body (refresh token + client secret) in `err.config`, and `JSON.parse` SyntaxErrors echo fragments of the unparseable token/credentials file. All auth-layer error logging now extracts only known-safe fields
- **auth:** fix `gcp-oauth.keys.json` project-root fallback resolution in the bundled build — the package root was computed assuming an unbundled `dist/auth/utils.js` layout, so on the actual bundled `dist/index.js` layout it resolved one directory too high and the fallback never matched, surfacing as "OAuth credentials not found" even with a valid keys file at the project root

## [2.3.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.2.0...v2.3.0) (2026-07-11)

Substantial internal refactor to introduce **multi-account support**: one running server can now hold OAuth credentials for several Google accounts (e.g. personal + Workspace) and route each tool call to the right identity. The change is strictly additive at the contract level — existing single-account users upgrade with no re-consent, no config changes, and no user-visible behavior change.

### Features

- **sheets:** add `ONE_OF_RANGE` to `addDataValidation` condition types, enabling dropdowns sourced from a cell range (Data validation → "Dropdown (from a range)"). Takes exactly one value — the source range in A1 notation; a leading `=` is added automatically if omitted. Lets dropdown option lists be maintained in one place (including a separate master spreadsheet via an `IMPORTRANGE` staging range)
- **calendar:** surface event `attachments` in `getCalendarEvent`/`getCalendarEvents` responses, and accept an `attachments` array (max 25) in `createCalendarEvent`/`updateCalendarEvent` (sets the `supportsAttachments` API flag). `updateCalendarEvent` now also preserves an event's existing attachments instead of silently dropping them when `attachments` is not supplied ([#110](https://github.com/piotr-agier/google-drive-mcp/issues/110))
- **auth:** support Workspace domain-wide delegation via `GOOGLE_DRIVE_MCP_SUBJECT`; `GOOGLE_DRIVE_MCP_SCOPES` is now honored in service-account mode ([#107](https://github.com/piotr-agier/google-drive-mcp/pull/107))
- **auth:** `authGetStatus` now verifies the *effective* Google identity the live Drive client is acting as (via Drive `about.get`) and reports the active auth mode (`oauth`/`service_account`/`external_token`) plus which override env vars are set. It warns when `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_DRIVE_MCP_ACCESS_TOKEN` in the environment cause a present `tokens.json` to be silently ignored (the same warning is now logged at startup). This makes "valid `auth/drive` token yet every call returns empty" misconfigurations diagnosable instead of invisible ([#137](https://github.com/piotr-agier/google-drive-mcp/issues/137))
- **resources:** add `GOOGLE_DRIVE_MCP_DISABLE_RESOURCES` env var (and `--no-resources` flag) to opt out of the MCP resource protocol (`gdrive:///` listing/reading); tools stay available. For tools-only clients or clients that hang enumerating a large Drive. The env var and `--no-resources[=<bool>]` accept `1/0`, `true/false`, `yes/no`, `on/off`; `--no-resources=false` re-enables resources, overriding a truthy env value ([#115](https://github.com/piotr-agier/google-drive-mcp/issues/115), [#128](https://github.com/piotr-agier/google-drive-mcp/pull/128))
- **drive/docs:** add `readTextFile` and extend `insertText`/`deleteRange` to edit raw `text/*` files (e.g. `text/plain`, `text/markdown`, `text/csv`) in place, alongside Google Docs. Text files are addressed by a 0-based Unicode code-point (character) offset, so edits and truncation stay safe for content containing emoji/astral characters; deleting a file's entire content correctly empties it; edits keep working under the `content-editor` (`drive.file`) scope; and `text/*` acceptance is consistent across `readTextFile`/`insertText`/`deleteRange`/`updateTextFile` (Google Docs indexes remain 1-based) ([#133](https://github.com/piotr-agier/google-drive-mcp/pull/133), [#141](https://github.com/piotr-agier/google-drive-mcp/pull/141))
- **sheets:** add column/row dimension tools — `setColumnWidth`, `setRowHeight`, `autoResizeColumns`, `autoResizeRows`, and `hideSheetDimension`/`showSheetDimension` — for resizing, auto-fitting, and hiding/showing columns and rows. Indices are 0-based and the interval is half-open `[start, end)`. Ranges are validated client-side (`start < end`, and `pixelSize >= 0` for width/height), so reversed or empty ranges are rejected up front with a clear message instead of an opaque Google API `400` ([#134](https://github.com/piotr-agier/google-drive-mcp/pull/134))
- **auth:** add `manage_accounts` tool with `list`, `add`, `remove`, and `set_default` actions for managing multiple connected Google accounts in a single process (local OAuth mode only; service-account and external-token modes remain single-identity).
- **auth:** add optional `account` parameter to every non-admin tool so a single call can be routed to a specific connected Google account. Resolution order is: explicit `account` → global default → sole-eligible account. Write tools refuse ambiguous resolution and point at `manage_accounts set_default`.
- **auth:** connect identity discovery — `manage_accounts add` requests `openid` and `userinfo.email` scopes so new accounts record their Google stable `sub` and email at consent time. Existing accounts are left untouched (no forced re-consent).
- **auth:** atomic-rename writes for `tokens.json` plus a process-wide write queue that serializes concurrent refreshes from different accounts.
- **auth:** per-alias refresh dedupe — N concurrent tool calls on the same account fire at most one refresh request to Google.

### Bug Fixes

- **drive/docs:** widen Shared Drive coverage on the remaining endpoints that omitted it — `getDocumentInfo` now passes `supportsAllDrives`, and `listFolder`, `listGoogleDocs`, `listGoogleSheets`, `bulkConvertFolderPdfs`, `authTestFileAccess`, and the `gdrive:///` resource listing now pass `corpora=allDrives` (alongside the flags they already set) so Shared Drive items surface consistently, matching `search` ([#137](https://github.com/piotr-agier/google-drive-mcp/issues/137))
- **drive:** `addPermission` no longer forces `emailAddress` for `type: "anyone"` and `type: "domain"` — the Drive API rejected the field for those principals, making both unusable. Added a `domain` parameter so domain-wide grants work, and per-type requirements are now validated up front (`emailAddress` for `user`/`group`, `domain` for `domain`, neither for `anyone`). Also added an optional `allowFileDiscovery` flag for `anyone`/`domain` grants — `false` (default) keeps a file link-only, `true` makes it discoverable in search ([#131](https://github.com/piotr-agier/google-drive-mcp/issues/131))
- **resources:** raise the `resources/list` page size from 10 to 1000 (Drive API max) so clients that eagerly enumerate the entire Drive (e.g. Gemini CLI) no longer hang during initialization ([#111](https://github.com/piotr-agier/google-drive-mcp/issues/111), [#128](https://github.com/piotr-agier/google-drive-mcp/pull/128))
- **docs:** honor `tabId` in `insertTable`, `editTableCell`, `insertSmartChip`, `createFootnote`, `applyTextStyle`/`formatGoogleDocText`, `applyParagraphStyle`/`formatGoogleDocParagraph`, and `createParagraphBullets` — these previously ignored `tabId` and silently edited the default tab of multi-tab documents while reporting success ([#114](https://github.com/piotr-agier/google-drive-mcp/issues/114), [#126](https://github.com/piotr-agier/google-drive-mcp/pull/126))
- **auth:** use loopback IP `127.0.0.1` instead of `localhost` for the OAuth callback redirect URI, matching the IPv4-only callback-server bind so the redirect resolves to the bound address on dual-stack hosts ([#124](https://github.com/piotr-agier/google-drive-mcp/pull/124)). Desktop-app OAuth clients (the recommended type) are unaffected; "Web application" clients that registered a `http://localhost:<port>` redirect must re-register it as `http://127.0.0.1:<port>` or auth fails with `redirect_uri_mismatch` — see [Troubleshooting](docs/troubleshooting.md)

### Performance Improvements

- **docs:** `applyParagraphStyle` with `textToFind` + `tabId` now resolves the matched range and its enclosing paragraph from a single `documents.get`, instead of two full unmasked `includeTabsContent` fetches of the same document. The non-tab path is unchanged ([#114](https://github.com/piotr-agier/google-drive-mcp/issues/114), [#127](https://github.com/piotr-agier/google-drive-mcp/pull/127))

### Token file format

`tokens.json` now uses a versioned v2 schema that keys multiple accounts by alias. The upgrade from v1 is **automatic** on first boot: the v2 file is written in place, the previous file is preserved as `tokens.json.v1-backup-<timestamp>` in case you need to roll back, and the migrated credentials are registered under the alias `default` (which is reserved — you cannot re-create it with `manage_accounts add`). The record is initially marked `pendingIdentity: true`; its email and Google `sub` populate the next time you re-consent.

Note: downgrading to 2.2.x or earlier after running 2.3.0+ requires manually restoring the `.v1-backup-*` file in place of the v2 `tokens.json`. The new file is not readable by older versions.

### Reserved aliases

`default`, `all`, `*`, `stdio`, `service-account`, `external-token`, `test` cannot be used with `manage_accounts add`.

### Known Limitations

- Cross-account read fanout (`account: string[]`) is planned but not yet shipped.
- The Streamable HTTP transport shares one active-default account across all sessions on the same process; per-session isolation is a planned follow-up.
- `manage_accounts remove` deletes local credentials but does not yet revoke the refresh token server-side — revoke manually via [Google Account Permissions](https://myaccount.google.com/permissions) if needed.

## [2.2.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.1.0...v2.2.0) (2026-04-20)

### Features

- **docs:** add optional `tabId` to `insertText`, `deleteRange`, `findAndReplaceInDoc`, and `updateGoogleDoc` for targeting specific tabs ([e2b5748](https://github.com/piotr-agier/google-drive-mcp/commit/e2b5748), [3bbf24f](https://github.com/piotr-agier/google-drive-mcp/commit/3bbf24f), [6d29b04](https://github.com/piotr-agier/google-drive-mcp/commit/6d29b04))
- **auth:** add `GOOGLE_DRIVE_MCP_AUTH_PORT` env var for configurable OAuth callback port ([95615b3](https://github.com/piotr-agier/google-drive-mcp/commit/95615b3), [f16fc3f](https://github.com/piotr-agier/google-drive-mcp/commit/f16fc3f))
- **drive:** add `emailMessage` support to `addPermission` and `shareFile` ([2fa3f52](https://github.com/piotr-agier/google-drive-mcp/commit/2fa3f52))

### Bug Fixes

- **docs:** fix `renameDocumentTab` ([e2b5748](https://github.com/piotr-agier/google-drive-mcp/commit/e2b5748))

## [2.1.0](https://github.com/piotr-agier/google-drive-mcp/compare/v2.0.2...v2.1.0) (2026-04-14)

### Features

- **slides:** add `insertSlidesImageFromUrl` and `insertSlidesLocalImage` tools ([8d7ae13](https://github.com/piotr-agier/google-drive-mcp/commit/8d7ae13))
- **slides:** add element management tools — move, delete, and inspect slide elements ([cb108df](https://github.com/piotr-agier/google-drive-mcp/commit/cb108df))

## [2.0.2](https://github.com/piotr-agier/google-drive-mcp/compare/v2.0.1...v2.0.2) (2026-04-04)

### Bug Fixes

- **docs:** use correct API field name for tab creation ([08caa89](https://github.com/piotr-agier/google-drive-mcp/commit/08caa89))
- use correct API field name for tab properties update ([8a67c75](https://github.com/piotr-agier/google-drive-mcp/commit/8a67c75))

## [2.0.1](https://github.com/piotr-agier/google-drive-mcp/compare/v2.0.0...v2.0.1) (2026-04-01)

### Bug Fixes

- **slides:** skip deleteText for empty speaker notes in Google Slides ([8f02fd1](https://github.com/piotr-agier/google-drive-mcp/commit/8f02fd1))

## [2.0.0](https://github.com/piotr-agier/google-drive-mcp/compare/v1.7.6...v2.0.0) (2026-03-28)

### Breaking Changes

- The server now supports two transport modes: **stdio** (default, unchanged) and **Streamable HTTP**. CLI arguments have been restructured to accommodate this — see [Configuration](docs/configuration.md) for details.

### Features

- **transport:** add Streamable HTTP transport mode (`--transport http`) with session management, SSE streaming, and configurable host/port ([f9aa097](https://github.com/piotr-agier/google-drive-mcp/commit/f9aa097))
- **auth:** support service account (`--service-account`) and external OAuth token (`--oauth-token`) authentication ([395ef05](https://github.com/piotr-agier/google-drive-mcp/commit/395ef05))

### Bug Fixes

- **transport:** add error handling to HTTP routes and extract shared route setup ([497e809](https://github.com/piotr-agier/google-drive-mcp/commit/497e809))
- **transport:** add session idle timeout, proper server cleanup, and security warning for non-localhost binding ([71ac0cb](https://github.com/piotr-agier/google-drive-mcp/commit/71ac0cb))

### Tests

- **transport:** add comprehensive HTTP transport and CLI argument tests ([03120c3](https://github.com/piotr-agier/google-drive-mcp/commit/03120c3))

## [1.7.6](https://github.com/piotr-agier/google-drive-mcp/compare/v1.7.5...v1.7.6) (2026-03-18)

### Features

- **docs:** add createFootnote tool ([fc0505a](https://github.com/piotr-agier/google-drive-mcp/commit/fc0505a))
- **docs:** extract tables and TOC in getGoogleDocContent ([5c97c4b](https://github.com/piotr-agier/google-drive-mcp/commit/5c97c4b))
- **docs:** extract inline elements in getGoogleDocContent ([7c7218e](https://github.com/piotr-agier/google-drive-mcp/commit/7c7218e))
- **drive:** add lockFile and unlockFile tools ([0a8b62b](https://github.com/piotr-agier/google-drive-mcp/commit/0a8b62b))
- **drive:** add createShortcut tool ([3b1efac](https://github.com/piotr-agier/google-drive-mcp/commit/3b1efac))

### Bug Fixes

- **docs,drive:** handle createFootnote partial failure, remove as-any casts ([329b8e3](https://github.com/piotr-agier/google-drive-mcp/commit/329b8e3))
- **drive:** unlockFile silently failed to remove content restriction ([efec828](https://github.com/piotr-agier/google-drive-mcp/commit/efec828))
- **docker,auth:** kill stale MCP process and simplify auth callback ([0dd0eba](https://github.com/piotr-agier/google-drive-mcp/commit/0dd0eba))
- **docs:** escape brackets in rich link titles and handle missing inlineObjects ([dfee405](https://github.com/piotr-agier/google-drive-mcp/commit/dfee405))
- **docker:** recreate container when image changes ([95e479b](https://github.com/piotr-agier/google-drive-mcp/commit/95e479b))

## [1.7.5](https://github.com/piotr-agier/google-drive-mcp/compare/v1.7.4...v1.7.5) (2026-03-14)

### Features

- **docker:** add wrapper script to reuse running container ([4945378](https://github.com/piotr-agier/google-drive-mcp/commit/4945378))

### Bug Fixes

- **docker:** improve wrapper script robustness and docs accuracy ([09e7bc9](https://github.com/piotr-agier/google-drive-mcp/commit/09e7bc9))
- **docker:** convert wrapper script line endings from CRLF to LF ([14659f1](https://github.com/piotr-agier/google-drive-mcp/commit/14659f1))
- **docs:** use $HOME instead of ~ in Docker volume mount examples ([ea2755f](https://github.com/piotr-agier/google-drive-mcp/commit/ea2755f))

## [1.7.4](https://github.com/piotr-agier/google-drive-mcp/compare/v1.7.3...v1.7.4) (2026-03-11)

### Bug Fixes

- **auth:** use stable config directory for credentials lookup ([50377ed](https://github.com/piotr-agier/google-drive-mcp/commit/50377ed))

### Refactors

- **auth:** remove dead code, surface parse errors, DRY config path ([661b4ce](https://github.com/piotr-agier/google-drive-mcp/commit/661b4ce))

## [1.7.3](https://github.com/piotr-agier/google-drive-mcp/compare/v1.7.2...v1.7.3) (2026-03-06)

### Features

- **docs:** add comment position context to listComments ([7a31c6f](https://github.com/piotr-agier/google-drive-mcp/commit/7a31c6f))

## [1.7.2](https://github.com/piotr-agier/google-drive-mcp/compare/v1.7.1...v1.7.2) (2026-03-03)

### Features

- **drive:** add convertToGoogleFormat param to uploadFile for native Google Workspace conversion ([4d7fc6d](https://github.com/piotr-agier/google-drive-mcp/commit/4d7fc6d))
- **docs:** add support for nested tabs to readGoogleDoc and getGoogleDocContent ([b0543a6](https://github.com/piotr-agier/google-drive-mcp/commit/b0543a6))

### Bug Fixes

- **sheets:** define nested items for appendSpreadsheetRows values schema ([75a71f5](https://github.com/piotr-agier/google-drive-mcp/commit/75a71f5))

## [1.7.1](https://github.com/piotr-agier/google-drive-mcp/compare/v1.7.0...v1.7.1) (2026-02-27)

### Features

- **search:** resolve folder paths in search results ([b10452b](https://github.com/piotr-agier/google-drive-mcp/commit/b10452b))
- **search:** add rawQuery for direct Google Drive API queries ([1da8349](https://github.com/piotr-agier/google-drive-mcp/commit/1da8349))

### Bug Fixes

- **search:** harden folder resolution and improve output consistency ([7384b8f](https://github.com/piotr-agier/google-drive-mcp/commit/7384b8f))
- remove authClearTokens and authSuggestScopePreset tools ([c373271](https://github.com/piotr-agier/google-drive-mcp/commit/c373271))

## [1.7.0](https://github.com/piotr-agier/google-drive-mcp/compare/v1.6.1...v1.7.0) (2026-02-26)

### Features

- add auth diagnostics and scope preset tools ([b5faad5](https://github.com/piotr-agier/google-drive-mcp/commit/b5faad5))
- add getRevisions and restoreRevision tools ([fc42683](https://github.com/piotr-agier/google-drive-mcp/commit/fc42683))

## [1.6.1](https://github.com/piotr-agier/google-drive-mcp/compare/v1.6.0...v1.6.1) (2026-02-26)

### Bug Fixes

- **search:** add corpora=allDrives so search returns Shared Drive results ([c0b9d6b](https://github.com/piotr-agier/google-drive-mcp/commit/c0b9d6b))

## [1.6.0](https://github.com/piotr-agier/google-drive-mcp/compare/v1.5.0...v1.6.0) (2026-02-26)

### Features

- add PDF ingestion and docs tab/chip transformation tools ([70ccca7](https://github.com/piotr-agier/google-drive-mcp/commit/70ccca7))
- implement real PDF splitting for uploadPdfWithSplit ([53f2b19](https://github.com/piotr-agier/google-drive-mcp/commit/53f2b19))

### Bug Fixes

- **insertSmartChip:** use correct Docs API structure, restrict to person chips only ([11e941a](https://github.com/piotr-agier/google-drive-mcp/commit/11e941a))

## [1.5.0](https://github.com/piotr-agier/google-drive-mcp/compare/v1.4.0...v1.5.0) (2026-02-26)

### Features

- add sheet governance and slide lifecycle tools ([9bc2563](https://github.com/piotr-agier/google-drive-mcp/commit/9bc2563))
- add sheets tab lifecycle and slides lifecycle/template helpers ([0af2a55](https://github.com/piotr-agier/google-drive-mcp/commit/0af2a55))
- add addSheet alias and slide thumbnail export ([d3c12d5](https://github.com/piotr-agier/google-drive-mcp/commit/d3c12d5))

### Bug Fixes

- **drive:** show inherited marker in listPermissions output ([b0423d2](https://github.com/piotr-agier/google-drive-mcp/commit/b0423d2))

## [1.4.0](https://github.com/piotr-agier/google-drive-mcp/compare/v1.3.3...v1.4.0) (2026-02-24)

### Features

- add docs formatting aliases, find/replace, and sharing permission tools ([464abcd](https://github.com/piotr-agier/google-drive-mcp/commit/464abcd))
- make shareFile idempotent by updating existing user permission ([f13e4c5](https://github.com/piotr-agier/google-drive-mcp/commit/f13e4c5))
- add removePermission by email, and find/replace dry-run ([1046046](https://github.com/piotr-agier/google-drive-mcp/commit/1046046))

## [1.3.3](https://github.com/piotr-agier/google-drive-mcp/compare/v1.3.2...v1.3.3) (2026-02-24)

### Bug Fixes

- **docs:** support multi-tab documents in readGoogleDoc ([cd46227](https://github.com/piotr-agier/google-drive-mcp/commit/cd46227))

## [1.3.2](https://github.com/piotr-agier/google-drive-mcp/compare/v1.3.1...v1.3.2) (2026-02-24)

### Features

- **drive:** add listSharedDrives tool ([dc1dd78](https://github.com/piotr-agier/google-drive-mcp/commit/dc1dd78))
- **auth:** allow OAuth scope override via env var ([45f42cb](https://github.com/piotr-agier/google-drive-mcp/commit/45f42cb))

### Bug Fixes

- **schema:** remove non-standard optional field from tool schemas ([943d71d](https://github.com/piotr-agier/google-drive-mcp/commit/943d71d))

## [1.3.1](https://github.com/piotr-agier/google-drive-mcp/compare/v1.3.0...v1.3.1) (2026-02-24)

### Bug Fixes

- CI/CD publishing fixes for npm OIDC trusted publishing ([160c0aa](https://github.com/piotr-agier/google-drive-mcp/commit/160c0aa))

## [1.3.0](https://github.com/piotr-agier/google-drive-mcp/compare/v1.2.0...v1.3.0) (2026-02-24)

### Features

- add includeFormatting option to getGoogleDocContent ([b30d6a0](https://github.com/piotr-agier/google-drive-mcp/commit/b30d6a0))
- add listComments pagination and multi-tab getGoogleDocContent ([a4992fc](https://github.com/piotr-agier/google-drive-mcp/commit/a4992fc))
- enrich fonts summary with sizes and styles per font ([f814577](https://github.com/piotr-agier/google-drive-mcp/commit/f814577))
- add 23 new tools for Calendar, Docs editing, Comments, Formatting ([baa8f6b](https://github.com/piotr-agier/google-drive-mcp/commit/baa8f6b))
- add 5 Phase 2 tools (Sheets management + copyFile) ([446f856](https://github.com/piotr-agier/google-drive-mcp/commit/446f856))
- add downloadFile tool ([95b70a5](https://github.com/piotr-agier/google-drive-mcp/commit/95b70a5))

### Bug Fixes

- bump @modelcontextprotocol/sdk to ^1.24.0 (CVE-2025-66414) ([4cf6024](https://github.com/piotr-agier/google-drive-mcp/commit/4cf6024))
- stop making uploaded images public by default in insertLocalImage ([a4d8df4](https://github.com/piotr-agier/google-drive-mcp/commit/a4d8df4))

## [1.2.0](https://github.com/piotr-agier/google-drive-mcp/compare/v1.1.2...v1.2.0) (2026-02-15)

### Features

- add uploadFile tool for binary file uploads ([4729309](https://github.com/piotr-agier/google-drive-mcp/commit/4729309))
- add Google Slides speaker notes support ([25b249e](https://github.com/piotr-agier/google-drive-mcp/commit/25b249e))
- add shared drives support to all Google Drive API operations ([d09caff](https://github.com/piotr-agier/google-drive-mcp/commit/d09caff))
- add valueInputOption parameter to createGoogleSheet and updateGoogleSheet ([77f56c7](https://github.com/piotr-agier/google-drive-mcp/commit/77f56c7))
- **search:** include file ID in search results ([68f031b](https://github.com/piotr-agier/google-drive-mcp/commit/68f031b))

## [1.1.2](https://github.com/piotr-agier/google-drive-mcp/releases/tag/v1.1.2) (2025-11-26)

### Features

- add pagination support to search tool ([b599b27](https://github.com/piotr-agier/google-drive-mcp/commit/b599b27))
- add comprehensive Google Sheets, Slides, and Docs formatting tools
- add Docker support with comprehensive documentation

### Bug Fixes

- fix sheet name parsing in Google Sheets formatting tools ([de693e5](https://github.com/piotr-agier/google-drive-mcp/commit/de693e5))
- fix 'Sheet not found' error for sheets with ID 0 ([c17fe97](https://github.com/piotr-agier/google-drive-mcp/commit/c17fe97))
