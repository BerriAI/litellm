import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import { findPackageRoot } from '../../src/auth/utils.js';

// Regression guard for the OAuth-keys project-root fallback. The build bundles
// to `dist/index.js`, so `getProjectRoot()` used to resolve two levels up from
// `dist/` — i.e. the package's PARENT — and the keys fallback path never
// matched. `findPackageRoot` must resolve the package root regardless of build
// layout. Note: the unbundled test build happens to make the old "up two"
// correct, which is exactly why the bug shipped — hence this fixture-based test
// that exercises the bundled layout explicitly.
test('findPackageRoot resolves the package root for both bundled and unbundled layouts', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gdmcp-pkgroot-'));
  try {
    fs.writeFileSync(path.join(root, 'package.json'), '{"name":"fixture"}');
    const distAuth = path.join(root, 'dist', 'auth');
    fs.mkdirSync(distAuth, { recursive: true });

    // Bundled: module at <root>/dist/index.js  ->  startDir = <root>/dist
    const bundledStart = path.join(root, 'dist');
    assert.equal(
      findPackageRoot(bundledStart),
      root,
      'bundled dist/index.js must resolve to the package root, not its parent',
    );
    // The old hard-coded "up two levels" would have returned the parent of root.
    assert.notEqual(
      findPackageRoot(bundledStart),
      path.resolve(bundledStart, '..', '..'),
      'must not regress to the package parent (the original bug)',
    );

    // Unbundled: module at <root>/dist/auth/utils.js  ->  startDir = <root>/dist/auth
    assert.equal(
      findPackageRoot(distAuth),
      root,
      'unbundled dist/auth/utils.js must also resolve to the package root',
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
