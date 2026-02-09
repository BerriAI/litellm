/**
 * Maps OpenClaw auth profiles to LiteLLM environment variable names.
 *
 * OpenClaw stores API keys in ~/.openclaw/agents/main/agent/auth-profiles.json
 * with the structure: { version, profiles: { "provider:profileId": { type, provider, key } } }
 *
 * This module reads those profiles and returns a Record<string, string>
 * of environment variable names to key values, suitable for passing
 * to the LiteLLM proxy process.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { homedir } from "node:os";

/** Mapping from OpenClaw provider name to LiteLLM env var */
const PROVIDER_ENV_MAP: Record<string, string> = {
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  google: "GOOGLE_API_KEY",
  deepseek: "DEEPSEEK_API_KEY",
  zai: "ZAI_API_KEY",
  xai: "XAI_API_KEY",
  moonshot: "MOONSHOT_API_KEY",
  minimax: "MINIMAX_API_KEY",
};

interface AuthProfileEntry {
  type: string;
  provider: string;
  key?: string;
}

interface AuthProfilesFile {
  version: number;
  profiles: Record<string, AuthProfileEntry>;
}

/**
 * Extract API keys from OpenClaw auth profiles.
 *
 * Reads `~/.openclaw/agents/main/agent/auth-profiles.json` and maps
 * any `type: "api_key"` profiles to their corresponding LiteLLM env var names.
 *
 * @returns Record of env var name -> API key value
 */
export async function extractApiKeys(): Promise<Record<string, string>> {
  const envVars: Record<string, string> = {};

  const profilePath = join(
    homedir(),
    ".openclaw",
    "agents",
    "main",
    "agent",
    "auth-profiles.json",
  );

  let data: AuthProfilesFile;
  try {
    const raw = await readFile(profilePath, "utf-8");
    data = JSON.parse(raw);
  } catch {
    // No auth profiles found — return empty
    return envVars;
  }

  if (!data.profiles || typeof data.profiles !== "object") {
    return envVars;
  }

  for (const [_profileId, profile] of Object.entries(data.profiles)) {
    if (profile.type !== "api_key" || !profile.key) {
      continue;
    }

    const envVar = PROVIDER_ENV_MAP[profile.provider];
    if (envVar) {
      envVars[envVar] = profile.key;
    }
  }

  return envVars;
}

/**
 * Returns the set of provider names that have API keys available.
 */
export function availableProviders(
  envVars: Record<string, string>,
): Set<string> {
  const providers = new Set<string>();

  for (const [provider, envVar] of Object.entries(PROVIDER_ENV_MAP)) {
    if (envVars[envVar]) {
      providers.add(provider);
    }
  }

  return providers;
}
