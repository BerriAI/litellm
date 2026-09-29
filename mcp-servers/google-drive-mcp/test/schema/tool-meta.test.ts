import assert from 'node:assert/strict';
import test from 'node:test';

import * as driveTools from '../../src/tools/drive.js';
import * as docsTools from '../../src/tools/docs.js';
import * as sheetsTools from '../../src/tools/sheets.js';
import * as slidesTools from '../../src/tools/slides.js';
import * as calendarTools from '../../src/tools/calendar.js';
import { ADMIN_TOOLS, TOOL_META } from '../../src/tools/toolMeta.js';

// ---------------------------------------------------------------------------
// TOOL_META completeness — bi-directional.
//
// Every registered tool must have a TOOL_META entry, and every TOOL_META entry
// must correspond to a registered tool. A missing entry would silently fall
// through to FALLBACK_META and skip scope enforcement — this test makes that
// drift fail loudly.
// ---------------------------------------------------------------------------

const registeredToolNames: string[] = [
  driveTools,
  docsTools,
  sheetsTools,
  slidesTools,
  calendarTools,
].flatMap((m) => m.toolDefinitions.map((d) => d.name));

test('every registered tool has a TOOL_META entry', () => {
  const missing = registeredToolNames.filter((n) => !(n in TOOL_META));
  assert.equal(
    missing.length,
    0,
    `Tools without TOOL_META entry: ${missing.join(', ')}`,
  );
});

test('no TOOL_META entry references an unregistered tool', () => {
  const registered = new Set(registeredToolNames);
  const stale = Object.keys(TOOL_META).filter((n) => !registered.has(n));
  assert.equal(stale.length, 0, `Stale TOOL_META entries: ${stale.join(', ')}`);
});

test('ADMIN_TOOLS matches the admin entries in TOOL_META', () => {
  const expectedAdmins = new Set(
    Object.entries(TOOL_META)
      .filter(([, m]) => m.opKind === 'admin')
      .map(([name]) => name),
  );
  assert.deepEqual([...ADMIN_TOOLS].sort(), [...expectedAdmins].sort());
});

test('non-admin tools declare at least one acceptable scope', () => {
  // A non-admin tool with an empty acceptableScopes list would pass any-of
  // scope filtering unconditionally — almost certainly a mistake. Admin tools
  // are exempt because they run on the default account regardless of scope.
  const violations = Object.entries(TOOL_META)
    .filter(([, m]) => m.opKind !== 'admin' && m.acceptableScopes.length === 0)
    .map(([name]) => name);
  assert.equal(
    violations.length,
    0,
    `Non-admin tools with empty acceptableScopes: ${violations.join(', ')}`,
  );
});

test('every acceptable scope is a full URL, not an alias', () => {
  // The resolver does exact matching; aliases like "drive" would silently
  // miss against stored scope strings like "https://www.googleapis.com/auth/drive".
  const violations: string[] = [];
  for (const [name, meta] of Object.entries(TOOL_META)) {
    for (const scope of meta.acceptableScopes) {
      if (!scope.startsWith('https://')) {
        violations.push(`${name}: "${scope}"`);
      }
    }
  }
  assert.equal(violations.length, 0, `Non-URL scopes: ${violations.join(', ')}`);
});

// ---------------------------------------------------------------------------
// Docs/Sheets/Slides accept the drive-family scopes (finding 7).
//
// The Docs/Sheets/Slides APIs also honor drive / drive.file / drive.readonly,
// so an account consented with `drive` must be eligible. drive.readonly can
// authorize reads but NOT writes, so it must appear on read tools only. A bare
// constant rename can't verify this directionality — these tests can.
// ---------------------------------------------------------------------------

const DRIVE = 'https://www.googleapis.com/auth/drive';
const DRIVE_READONLY = 'https://www.googleapis.com/auth/drive.readonly';

const PRODUCT_TOOLS: Array<{ read: string; write: string }> = [
  { read: 'readGoogleDoc', write: 'createGoogleDoc' },
  { read: 'getGoogleSheetContent', write: 'createGoogleSheet' },
  { read: 'getGoogleSlidesContent', write: 'createGoogleSlides' },
];

for (const { read, write } of PRODUCT_TOOLS) {
  test(`${read} (read) accepts full drive AND drive.readonly`, () => {
    const scopes = TOOL_META[read].acceptableScopes;
    assert.ok(scopes.includes(DRIVE), `${read} should accept ${DRIVE}`);
    assert.ok(scopes.includes(DRIVE_READONLY), `${read} should accept ${DRIVE_READONLY}`);
  });

  test(`${write} (write) accepts full drive but NOT drive.readonly`, () => {
    const scopes = TOOL_META[write].acceptableScopes;
    assert.ok(scopes.includes(DRIVE), `${write} should accept ${DRIVE}`);
    assert.ok(
      !scopes.includes(DRIVE_READONLY),
      `${write} must not accept read-only ${DRIVE_READONLY}`,
    );
  });
}
