#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { withRetry } from '../src/utils/retry.core.js';

const registryBaseUrl = (process.env.MCP_REGISTRY_URL || 'https://registry.modelcontextprotocol.io').replace(/\/$/, '');
const command = process.argv[2];
const serverFile = resolve(process.cwd(), process.argv[3] || 'server.json');

// Shared retry policy for one-shot registry requests. Mirrors the app defaults
// (see src/utils/retry.core.js): ~3 attempts with a 15s per-attempt timeout.
const requestConfig = { apiTimeout: 15_000, retryMax: 2, retryBaseDelay: 1_000 };

function usage() {
  console.error('Usage: node scripts/registry-metadata.js <validate|verify> [server.json]');
}

function sleep(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

async function readServerMetadata() {
  try {
    return JSON.parse(await readFile(serverFile, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to read ${serverFile}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

// A registry request is "unreachable" (a transient/transport problem rather than
// a rejected payload) when it has no HTTP status (network/timeout/abort) or a
// status that signals server-side/transient trouble (429 or 5xx). Genuine 4xx
// client errors — a malformed request or wrong path — are NOT unreachable.
function isUnreachable(error) {
  const status = error?.status ?? error?.response?.status;
  if (typeof status === 'number') return status === 429 || status >= 500;
  return true;
}

async function requestJson(url, options, label) {
  try {
    return await withRetry(
      async (signal) => {
        let response;
        try {
          response = await fetch(url, { ...options, signal });
        } catch (error) {
          // Transport-level failure (DNS/connection/abort). Give it a retryable
          // code so withRetry treats it like a transient error.
          if (typeof error.code !== 'string') {
            error.code = error?.cause?.code || 'ECONNRESET';
          }
          throw error;
        }

        const body = await response.text();
        let parsed;
        try {
          parsed = JSON.parse(body);
        } catch {
          const error = new Error(`Registry returned non-JSON response (${response.status}): ${body.slice(0, 500)}`);
          error.status = response.status;
          throw error;
        }

        if (!response.ok) {
          // withRetry retries 429/503/504 and gives up immediately on other 4xx/5xx.
          const error = new Error(`Registry request failed (${response.status}): ${JSON.stringify(parsed)}`);
          error.status = response.status;
          throw error;
        }

        return parsed;
      },
      requestConfig,
      label,
    );
  } catch (error) {
    if (isUnreachable(error)) error.unreachable = true;
    throw error;
  }
}

function formatIssue(issue) {
  const path = issue.path ? `${issue.path}: ` : '';
  const severity = issue.severity ? `[${issue.severity}] ` : '';
  return `${severity}${path}${issue.message || JSON.stringify(issue)}`;
}

async function validate(server) {
  const result = await requestJson(
    `${registryBaseUrl}/v0.1/validate`,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(server),
    },
    'registry:validate',
  );

  if (!result.valid) {
    const issues = Array.isArray(result.issues) ? result.issues.map(formatIssue).join('\n') : JSON.stringify(result);
    throw new Error(`MCP Registry validation failed:\n${issues}`);
  }

  console.log(`MCP Registry validation passed for ${server.name}@${server.version}.`);
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// True when every field declared in `expected` is present in `actual` with an
// equal value. Object key order and any extra fields the registry adds (e.g.
// registryBaseUrl, status metadata) are ignored — only the metadata we own is
// asserted, so re-serialization by the registry does not read as a mismatch.
export function matchesExpected(expected, actual) {
  if (Array.isArray(expected)) {
    return (
      Array.isArray(actual) &&
      actual.length === expected.length &&
      expected.every((item, index) => matchesExpected(item, actual[index]))
    );
  }
  if (isPlainObject(expected)) {
    return isPlainObject(actual) && Object.keys(expected).every((key) => matchesExpected(expected[key], actual[key]));
  }
  return expected === actual;
}

export function comparePublishedRecord(expected, actual) {
  const differences = [];
  const compare = (label, expectedValue, actualValue) => {
    if (!matchesExpected(expectedValue, actualValue)) {
      differences.push(`${label}: expected ${JSON.stringify(expectedValue)}, got ${JSON.stringify(actualValue)}`);
    }
  };

  compare('title', expected.title, actual.title);
  compare('description', expected.description, actual.description);
  compare('repository', expected.repository, actual.repository);
  compare('websiteUrl', expected.websiteUrl, actual.websiteUrl);
  compare('packages', expected.packages, actual.packages);
  return differences;
}

async function verify(server) {
  const url = new URL(`${registryBaseUrl}/v0.1/servers`);
  url.searchParams.set('search', server.name);
  url.searchParams.set('version', server.version);
  const attempts = Math.max(1, Number.parseInt(process.env.MCP_REGISTRY_VERIFY_ATTEMPTS || '1', 10) || 1);

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const result = await requestJson(url, { method: 'GET' }, 'registry:verify');
    const record = result.servers?.find(
      (entry) => entry.server?.name === server.name && entry.server?.version === server.version,
    )?.server;

    if (record) {
      const differences = comparePublishedRecord(server, record);
      if (differences.length > 0) {
        throw new Error(`Published MCP Registry record does not match server.json:\n${differences.join('\n')}`);
      }
      console.log(`Verified published MCP Registry record ${server.name}@${server.version}.`);
      return;
    }

    if (attempt < attempts) await sleep(attempt * 2_000);
  }

  console.error(`MCP Registry record ${server.name}@${server.version} was not found.`);
  process.exitCode = 2;
}

const isMain = Boolean(process.argv[1]) && resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMain) {
  try {
    if (command !== 'validate' && command !== 'verify') {
      usage();
      process.exitCode = 1;
    } else {
      const server = await readServerMetadata();
      if (command === 'validate') await validate(server);
      else await verify(server);
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    // 3 = registry unreachable (transient); 1 = genuine failure (invalid
    // metadata, record mismatch, bad command). verify() sets 2 for not-found.
    process.exitCode = error?.unreachable ? 3 : 1;
  }
}
