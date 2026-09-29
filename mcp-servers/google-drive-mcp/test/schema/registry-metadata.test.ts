import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it } from 'node:test';

const readJson = (file: string) => JSON.parse(readFileSync(resolve(process.cwd(), file), 'utf8'));
const packageJson = readJson('package.json');
const packageLock = readJson('package-lock.json');
const serverJson = readJson('server.json');
const npmPackage = serverJson.packages.find((entry: any) => entry.registryType === 'npm');

describe('MCP Registry metadata', () => {
  it('uses the package mcpName as its stable Registry identity', () => {
    assert.equal(serverJson.name, 'io.github.piotr-agier/google-drive-mcp');
    assert.equal(packageJson.mcpName, serverJson.name);
  });

  it('keeps every released version in lockstep', () => {
    assert.equal(serverJson.version, packageJson.version);
    assert.equal(npmPackage.version, packageJson.version);
    assert.equal(packageLock.version, packageJson.version);
    assert.equal(packageLock.packages[''].version, packageJson.version);
  });

  it('advertises the published npm package over stdio', () => {
    assert.equal(npmPackage.identifier, packageJson.name);
    assert.equal(npmPackage.runtimeHint, 'npx');
    assert.deepEqual(npmPackage.transport, { type: 'stdio' });
    assert.equal(serverJson.packages.length, 1);
  });

  it('points discovery metadata at maintained project URLs', () => {
    assert.deepEqual(serverJson.repository, {
      url: 'https://github.com/piotr-agier/google-drive-mcp',
      source: 'github',
      id: '1028345101',
    });
    assert.equal(serverJson.websiteUrl, 'https://github.com/piotr-agier/google-drive-mcp#readme');
    assert.equal(packageJson.homepage, serverJson.websiteUrl);
    assert.equal(packageJson.repository.url, 'git+https://github.com/piotr-agier/google-drive-mcp.git');
  });

  it('keeps the description identical across npm and Registry metadata', () => {
    assert.equal(packageJson.description, serverJson.description);
  });

  it('prompts Registry installers for the OAuth credentials file', () => {
    assert.deepEqual(npmPackage.environmentVariables, [
      {
        name: 'GOOGLE_DRIVE_OAUTH_CREDENTIALS',
        description: 'Path to a Google OAuth 2.0 Desktop app credentials JSON file.',
        format: 'filepath',
        isRequired: true,
        isSecret: true,
      },
    ]);
  });
});
