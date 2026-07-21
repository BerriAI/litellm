import { webcrypto } from "node:crypto";
import { beforeAll, describe, expect, it } from "vitest";
import { decryptSecret, encryptSecret } from "./crypto";

beforeAll(() => {
  if (typeof globalThis.crypto?.subtle === "undefined") {
    Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });
  }
});

describe("secure share crypto", () => {
  it("round-trips a secret with the correct password", async () => {
    const secret = "sk-super-secret-value-123";
    const payload = await encryptSecret(secret, "correct horse battery staple");

    expect(payload.ciphertext).not.toContain(secret);
    await expect(decryptSecret(payload, "correct horse battery staple")).resolves.toBe(secret);
  });

  it("produces a fresh salt and iv on every call", async () => {
    const first = await encryptSecret("value", "password123");
    const second = await encryptSecret("value", "password123");

    expect(first.salt).not.toBe(second.salt);
    expect(first.iv).not.toBe(second.iv);
    expect(first.ciphertext).not.toBe(second.ciphertext);
  });

  it("rejects decryption with the wrong password", async () => {
    const payload = await encryptSecret("value", "password123");
    await expect(decryptSecret(payload, "wrong-password")).rejects.toThrow();
  });

  it("rejects decryption when the ciphertext is tampered with", async () => {
    const payload = await encryptSecret("value", "password123");
    const tampered = { ...payload, ciphertext: btoa("tampered-ciphertext-bytes") };
    await expect(decryptSecret(tampered, "password123")).rejects.toThrow();
  });
});
