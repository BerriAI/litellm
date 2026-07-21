const PBKDF2_ITERATIONS = 210_000;
const SALT_BYTES = 16;
const IV_BYTES = 12;
const AES_KEY_BITS = 256;

export interface EncryptedPayload {
  ciphertext: string;
  salt: string;
  iv: string;
}

function toBase64(bytes: Uint8Array<ArrayBuffer>): string {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

function fromBase64(value: string): Uint8Array<ArrayBuffer> {
  const binary = atob(value);
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

async function deriveKey(password: string, salt: Uint8Array<ArrayBuffer>): Promise<CryptoKey> {
  const keyMaterial = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, [
    "deriveKey",
  ]);
  const kdfParams: Pbkdf2Params = { name: "PBKDF2", salt, iterations: PBKDF2_ITERATIONS, hash: "SHA-256" };
  return crypto.subtle.deriveKey(kdfParams, keyMaterial, { name: "AES-GCM", length: AES_KEY_BITS }, false, [
    "encrypt",
    "decrypt",
  ]);
}

export async function encryptSecret(plaintext: string, password: string): Promise<EncryptedPayload> {
  const salt = crypto.getRandomValues(new Uint8Array(SALT_BYTES));
  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const key = await deriveKey(password, salt);
  const cipher = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, new TextEncoder().encode(plaintext));
  return {
    ciphertext: toBase64(new Uint8Array(cipher)),
    salt: toBase64(salt),
    iv: toBase64(iv),
  };
}

export async function decryptSecret(payload: EncryptedPayload, password: string): Promise<string> {
  const salt = fromBase64(payload.salt);
  const iv = fromBase64(payload.iv);
  const key = await deriveKey(password, salt);
  const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, fromBase64(payload.ciphertext));
  return new TextDecoder().decode(plain);
}
