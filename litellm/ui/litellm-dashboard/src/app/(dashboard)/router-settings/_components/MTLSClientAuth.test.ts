import { describe, expect, it } from "vitest";
import { mtlsClientAuthIsValid } from "./MTLSClientAuth";

describe("certificate approval validation", () => {
  it("allows disabled approval and a deliberately empty allowlist", () => {
    expect(mtlsClientAuthIsValid(null)).toBe(true);
    expect(mtlsClientAuthIsValid({ allowed_clients: [] })).toBe(true);
  });

  it.each([
    { allowed_clients: [{ issuer: "", subject: "CN=worker" }] },
    { allowed_clients: [{ issuer: "CN=CA", subject: "" }] },
    { issuer_header: "bad header", allowed_clients: [] },
    { subject_header: "", allowed_clients: [] },
  ])("refuses incomplete clients and invalid header names", (config) => {
    expect(mtlsClientAuthIsValid(config)).toBe(false);
  });
});
