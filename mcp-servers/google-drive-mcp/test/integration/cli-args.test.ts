import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const DIST_INDEX = join(process.cwd(), 'dist', 'index.js');
const PKG = JSON.parse(readFileSync(join(process.cwd(), 'package.json'), 'utf8'));

interface RunResult {
  stdout: string;
  stderr: string;
  exitCode: number | null;
}

/**
 * Spawn the built CLI and wait until it exits on its own — or, for commands
 * that keep serving (`until` given), until that pattern shows up on stderr,
 * then kill it. The generous deadline replaces the old fixed 3s spawnSync
 * timeout, which node startup alone could blow through when the whole suite
 * runs in parallel; the pattern-match exit keeps the common case fast.
 */
function run(args: string[], env: Record<string, string> = {}, until?: RegExp): Promise<RunResult> {
  // Remove MCP_TESTING so main() actually runs in the subprocess
  const { MCP_TESTING: _, ...cleanEnv } = process.env;
  return new Promise((resolve) => {
    const child = spawn('node', [DIST_INDEX, ...args], {
      env: { ...cleanEnv, ...env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    let done = false;
    const finish = (exitCode: number | null) => {
      if (done) return;
      done = true;
      clearTimeout(deadline);
      resolve({ stdout, stderr, exitCode });
    };
    const deadline = setTimeout(() => {
      child.kill('SIGKILL');
      finish(null);
    }, 15_000);
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
      if (until && until.test(stderr)) {
        child.kill('SIGKILL');
        finish(null);
      }
    });
    child.on('exit', (code) => finish(code));
    child.on('error', () => finish(null));
  });
}

describe('CLI argument parsing', () => {
  it('--transport bogus exits with error', async () => {
    const result = await run(['--transport', 'bogus']);
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /Invalid transport/);
  });

  it('--port 0 exits with error', async () => {
    const result = await run(['--port', '0', '--transport', 'http']);
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /Invalid port/);
  });

  it('--port 99999 exits with error', async () => {
    const result = await run(['--port', '99999', '--transport', 'http']);
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /Invalid port/);
  });

  it('--port abc exits with error', async () => {
    const result = await run(['--port', 'abc', '--transport', 'http']);
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /Invalid port/);
  });

  it('default transport is stdio', async () => {
    // stdio mode blocks on stdin — killed once the banner shows
    const result = await run([], {}, /\(stdio\)/);
    assert.match(result.stderr, /\(stdio\)/);
  });

  it('--transport http starts HTTP', async () => {
    const result = await run(['--transport', 'http', '--port', '18923'], {}, /\(HTTP/i);
    assert.match(result.stderr, /\(HTTP/i);
  });

  it('env var fallback works', async () => {
    const result = await run([], { MCP_TRANSPORT: 'http', MCP_HTTP_PORT: '18924' }, /\(HTTP/i);
    assert.match(result.stderr, /\(HTTP/i);
  });

  it('CLI flags override env vars', async () => {
    const result = await run(['--transport', 'stdio'], { MCP_TRANSPORT: 'http' }, /\(stdio\)/);
    assert.match(result.stderr, /\(stdio\)/);
  });

  it('--version prints version and exits', async () => {
    const result = await run(['--version']);
    assert.equal(result.exitCode, 0);
    assert.ok(result.stdout.includes(PKG.version), `expected version ${PKG.version} in: ${result.stdout}`);
  });

  it('--help prints usage and exits', async () => {
    const result = await run(['--help']);
    assert.equal(result.exitCode, 0);
    assert.ok(
      result.stdout.includes('Usage') || result.stdout.includes('transport'),
      `expected usage info in: ${result.stdout}`,
    );
  });
});

describe('Team mode CLI validation', () => {
  it('--team without the HTTP transport exits with error', async () => {
    const result = await run(['--team']);
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /requires the HTTP transport/);
  });

  it('MCP_TEAM_MODE env var behaves like --team', async () => {
    const result = await run([], { MCP_TEAM_MODE: '1' });
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /requires the HTTP transport/);
  });

  it('MCP_TEAM_MODE does not block utility commands like version', async () => {
    // The transport guard applies only to server-start; version/help/auth must
    // still run with MCP_TEAM_MODE set (they default to the stdio transport).
    const result = await run(['--version'], { MCP_TEAM_MODE: '1' });
    assert.equal(result.exitCode, 0);
    assert.ok(result.stdout.includes(PKG.version), `expected version in: ${result.stdout}`);
  });

  it('a malformed credentials file fails with an actionable, secret-free error', async () => {
    const secret = 'TOPSECRET-client-secret-DO-NOT-LEAK-abcdef';
    const dir = mkdtempSync(join(tmpdir(), 'gdrive-mcp-cli-'));
    const keysPath = join(dir, 'gcp-oauth.keys.json');
    // Malformed JSON that embeds the secret next to the syntax error, so an
    // unsanitized JSON.parse SyntaxError message would echo it.
    writeFileSync(keysPath, `{"web":{"client_secret":"${secret}" OOPS`);
    try {
      const result = await run(
        ['--team', '--transport', 'http', '--port', '18930', '--issuer-url', 'http://127.0.0.1:18930'],
        {
          MCP_TEAM_STORE: 'memory',
          GOOGLE_DRIVE_OAUTH_CREDENTIALS: keysPath,
          XDG_CONFIG_HOME: '/nonexistent',
        },
      );
      assert.equal(result.exitCode, 1);
      const combined = result.stdout + result.stderr;
      assert.ok(!combined.includes(secret), `error output must not leak the secret: ${combined}`);
      assert.match(result.stderr, /credentials file|invalid JSON|Failed to start/i);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('--team without an issuer URL exits with error', async () => {
    const result = await run(['--team', '--transport', 'http', '--port', '18925']);
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /issuer URL/);
  });

  it('--team rejects a plain-http non-localhost issuer', async () => {
    const result = await run([
      '--team', '--transport', 'http', '--port', '18926',
      '--issuer-url', 'http://drive-mcp.example.com',
    ]);
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /must use https/);
  });

  it('--team is incompatible with service-account mode', async () => {
    const result = await run(
      ['--team', '--transport', 'http', '--port', '18927', '--issuer-url', 'http://127.0.0.1:18927'],
      { GOOGLE_APPLICATION_CREDENTIALS: '/tmp/sa.json' },
    );
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /incompatible with service-account mode/);
  });

  it('--team without web-client credentials exits with the setup hint', async () => {
    const result = await run(
      ['--team', '--transport', 'http', '--port', '18928', '--issuer-url', 'http://127.0.0.1:18928'],
      {
        MCP_TEAM_STORE: 'memory',
        // Point credential discovery at an empty directory.
        GOOGLE_DRIVE_OAUTH_CREDENTIALS: '/nonexistent/gcp-oauth.keys.json',
        XDG_CONFIG_HOME: '/nonexistent',
      },
    );
    assert.equal(result.exitCode, 1);
    assert.match(result.stderr, /Web application/);
  });

  it('--team boots with env-pair credentials and prints the derived URLs', async () => {
    const result = await run(
      ['--team', '--transport', 'http', '--port', '18929', '--issuer-url', 'http://127.0.0.1:18929'],
      {
        MCP_TEAM_STORE: 'memory',
        GOOGLE_DRIVE_MCP_CLIENT_ID: 'cid.apps.googleusercontent.com',
        GOOGLE_DRIVE_MCP_CLIENT_SECRET: 'csecret',
      },
      /register this redirect URI/,
    );
    assert.match(result.stderr, /Team mode enabled/);
    assert.match(result.stderr, /register this redirect URI/);
    assert.match(result.stderr, /oauth\/google\/callback/);
  });
});
