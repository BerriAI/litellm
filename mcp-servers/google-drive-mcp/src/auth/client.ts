import { OAuth2Client } from 'google-auth-library';
import * as fs from 'fs/promises';
import { describeErrorForLog, getKeysFilePaths, generateCredentialsErrorMessage, OAuthCredentials } from './utils.js';

function parseCredentialsFile(keys: Record<string, unknown>): OAuthCredentials {
  if (keys.installed) {
    const { client_id, client_secret, redirect_uris } = keys.installed as OAuthCredentials;
    return { client_id, client_secret, redirect_uris };
  } else if (keys.web) {
    const { client_id, client_secret, redirect_uris } = keys.web as OAuthCredentials;
    return { client_id, client_secret, redirect_uris };
  } else if (keys.client_id) {
    return {
      client_id: keys.client_id as string,
      client_secret: keys.client_secret as string | undefined,
      redirect_uris: (keys.redirect_uris as string[] | undefined) || ['http://127.0.0.1:3000/oauth2callback']
    };
  } else {
    throw new Error('Invalid credentials file format. Expected either "installed", "web" object or direct client_id field.');
  }
}

async function loadCredentialsFromFile(): Promise<OAuthCredentials> {
  const paths = getKeysFilePaths();

  for (const keysPath of paths) {
    try {
      const keysContent = await fs.readFile(keysPath, "utf-8");
      const keys = JSON.parse(keysContent);
      return parseCredentialsFile(keys);
    } catch (err: unknown) {
      // Re-throw parse/validation errors so the user gets actionable feedback
      if (err instanceof SyntaxError ||
          (err instanceof Error && err.message.includes('Invalid credentials'))) {
        // describeErrorForLog collapses a JSON.parse SyntaxError to a constant
        // rather than echoing file fragments (which may include the client_secret).
        throw new Error(`Invalid credentials file at ${keysPath}: ${describeErrorForLog(err)}`);
      }
      // File not found — try next path
    }
  }

  throw new Error(`Credentials file not found. Searched: ${paths.join(', ')}`);
}

async function loadCredentialsWithFallback(): Promise<OAuthCredentials> {
  try {
    return await loadCredentialsFromFile();
  } catch (fileError) {
    // Check for legacy client_secret.json
    const legacyPath = process.env.GOOGLE_CLIENT_SECRET_PATH || 'client_secret.json';
    try {
      const legacyContent = await fs.readFile(legacyPath, 'utf-8');
      const legacyKeys = JSON.parse(legacyContent);
      console.error('Warning: Using legacy client_secret.json. Please migrate to gcp-oauth.keys.json');
      
      if (legacyKeys.installed) {
        return legacyKeys.installed;
      } else if (legacyKeys.web) {
        return legacyKeys.web;
      } else {
        throw new Error('Invalid legacy credentials format');
      }
    } catch (_legacyError) {
      // Generate helpful error message
      const errorMessage = generateCredentialsErrorMessage();
      throw new Error(`${errorMessage}\n\nOriginal error: ${fileError instanceof Error ? fileError.message : fileError}`);
    }
  }
}

export async function initializeOAuth2Client(): Promise<OAuth2Client> {
  try {
    const credentials = await loadCredentialsWithFallback();
    
    // Use the first redirect URI as the default for the base client
    return new OAuth2Client({
      clientId: credentials.client_id,
      clientSecret: credentials.client_secret || undefined,
      redirectUri: credentials.redirect_uris?.[0] || 'http://127.0.0.1:3000/oauth2callback',
    });
  } catch (error) {
    throw new Error(`Error loading OAuth keys: ${error instanceof Error ? error.message : error}`);
  }
}

/**
 * Load web-application OAuth client credentials for team mode.
 *
 * Team mode's Google callback is a public URL, and Google restricts
 * desktop-type ("installed") clients to loopback redirect URIs — so a
 * "Web application" client is required. The env pair takes precedence (suits
 * secret-manager injection on Cloud Run); otherwise the keys file must carry
 * a `web` section (or the direct client_id form, where the type is the
 * operator's responsibility). An installed-only keys file fails with an
 * actionable error naming the redirect URI to register.
 */
export async function loadWebCredentials(
  redirectUri: string,
): Promise<{ client_id: string; client_secret: string }> {
  const envId = process.env.GOOGLE_DRIVE_MCP_CLIENT_ID;
  const envSecret = process.env.GOOGLE_DRIVE_MCP_CLIENT_SECRET;
  if (envId && envSecret) {
    return { client_id: envId, client_secret: envSecret };
  }

  const webClientHint =
    `Team mode requires an OAuth client of type "Web application". Create one in ` +
    `Google Cloud Console (APIs & Services → Credentials), add\n  ${redirectUri}\n` +
    `as an authorized redirect URI, and provide it via gcp-oauth.keys.json (the "web" ` +
    `form) or the GOOGLE_DRIVE_MCP_CLIENT_ID / GOOGLE_DRIVE_MCP_CLIENT_SECRET env pair.`;

  const paths = getKeysFilePaths();
  for (const keysPath of paths) {
    let keys: Record<string, unknown>;
    try {
      keys = JSON.parse(await fs.readFile(keysPath, 'utf-8'));
    } catch (err) {
      if (err instanceof SyntaxError) {
        // Do not interpolate the raw SyntaxError message: in Node it echoes a
        // snippet of the unparseable file, which for gcp-oauth.keys.json is the
        // client_secret. describeErrorForLog returns a constant for SyntaxError.
        throw new Error(`Invalid credentials file at ${keysPath}: ${describeErrorForLog(err)}`);
      }
      continue; // not found — try the next path
    }
    const web = keys.web as OAuthCredentials | undefined;
    if (web?.client_id && web.client_secret) {
      return { client_id: web.client_id, client_secret: web.client_secret };
    }
    if (keys.client_id && keys.client_secret) {
      // Direct form carries no type marker; the operator owns the client type.
      return {
        client_id: keys.client_id as string,
        client_secret: keys.client_secret as string,
      };
    }
    if (keys.installed) {
      throw new Error(
        `The credentials file at ${keysPath} contains a desktop-type ("installed") OAuth ` +
          `client, which Google restricts to loopback redirect URIs.\n${webClientHint}`,
      );
    }
    throw new Error(
      `The credentials file at ${keysPath} has no usable "web" client entry.\n${webClientHint}`,
    );
  }
  throw new Error(
    `No team-mode OAuth client credentials found (searched: ${paths.join(', ')}).\n${webClientHint}`,
  );
}

export async function loadCredentials(): Promise<{ client_id: string; client_secret?: string }> {
  try {
    const credentials = await loadCredentialsWithFallback();
    
    if (!credentials.client_id) {
        throw new Error('Client ID missing in credentials.');
    }
    return {
      client_id: credentials.client_id,
      client_secret: credentials.client_secret
    };
  } catch (error) {
    throw new Error(`Error loading credentials: ${error instanceof Error ? error.message : error}`);
  }
}