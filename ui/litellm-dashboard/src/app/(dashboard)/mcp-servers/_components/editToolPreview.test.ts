import { describe, expect, it } from "vitest";
import { getEditToolPreview } from "./editToolPreview";

const saved = {
  url: "https://example.com/mcp",
  transport: "http",
  auth_type: "basic",
  static_headers: [{ header: "X-Tenant", value: "original" }],
};

describe("getEditToolPreview", () => {
  it("keeps saved discovery for unchanged settings and unrelated edits", () => {
    expect(getEditToolPreview({ ...saved, server_name: "renamed" }, saved)).toEqual({ kind: "saved" });
  });

  it("previews URL changes with the existing server credential left for server-side inheritance", () => {
    expect(getEditToolPreview({ ...saved, url: "https://example.com/corrected-mcp" }, saved)).toEqual({
      kind: "preview",
      config: {
        url: "https://example.com/corrected-mcp",
        transport: "http",
        auth_type: "basic",
        static_headers: { "X-Tenant": "original" },
        credentials: undefined,
      },
    });
  });

  it.each(["https://other.example/mcp", "http://example.com/mcp", "https://example.com:8443/mcp"])(
    "requires explicit credentials for a changed origin: %s",
    (url) => {
      expect(getEditToolPreview({ ...saved, url, static_headers: [] }, saved)).toEqual({
        kind: "incomplete",
        message: expect.stringContaining("origin changed"),
      });
      const explicit = { ...saved, url, static_headers: [], credentials: { auth_value: "new:secret" } };
      expect(getEditToolPreview(explicit, saved).kind).toBe("preview");
    },
  );

  it("does not automatically send saved static headers to a new origin", () => {
    expect(getEditToolPreview({ ...saved, url: "https://other.example/mcp", auth_type: "none" }, saved).kind).toBe(
      "incomplete",
    );
  });

  it("uses edited static headers and only the static auth value", () => {
    expect(
      getEditToolPreview(
        {
          ...saved,
          static_headers: [{ header: "X-Tenant", value: "corrected" }],
          credentials: { auth_value: "user:password", access_token: "old-oauth-token", client_secret: "old-client" },
        },
        saved,
      ),
    ).toEqual({
      kind: "preview",
      config: {
        url: saved.url,
        transport: "http",
        auth_type: "basic",
        static_headers: { "X-Tenant": "corrected" },
        credentials: { auth_value: "user:password" },
      },
    });
  });

  it.each(["", "https://", "file:///tmp/server"])("does not connect to an incomplete or unsupported URL: %s", (url) => {
    expect(getEditToolPreview({ ...saved, url }, saved)).toEqual({ kind: "incomplete" });
  });

  it("waits for a static header value before connecting", () => {
    const values = { ...saved, static_headers: [{ header: "X-Tenant", value: "" }] };
    expect(getEditToolPreview(values, saved)).toEqual({ kind: "incomplete" });
  });

  it("waits for credentials when switching from None to Basic Auth", () => {
    expect(getEditToolPreview(saved, { ...saved, auth_type: "none" })).toEqual({ kind: "incomplete" });
  });

  it("does not forward old credentials when switching to None", () => {
    const result = getEditToolPreview(
      { ...saved, auth_type: "none", credentials: { auth_value: "old-secret" } },
      saved,
    );
    expect(result.kind).toBe("preview");
    if (result.kind === "preview") expect(result.config.credentials).toBeUndefined();
  });

  it.each(["oauth2", "true_passthrough", "oauth_delegate", "oauth2_token_exchange", "oauth2_id_jag", "aws_sigv4"])(
    "preserves the existing discovery path for %s",
    (auth_type) => {
      expect(getEditToolPreview({ ...saved, auth_type, url: "https://changed.example/mcp" }, saved)).toEqual({
        kind: "saved",
      });
    },
  );

  it("keeps stdio and OpenAPI on their existing discovery path", () => {
    for (const transport of ["stdio", "openapi"]) {
      expect(getEditToolPreview({ ...saved, transport }, saved)).toEqual({ kind: "saved" });
    }
  });

  describe("secret headers", () => {
    const savedCustom = { ...saved, auth_type: "secret_headers", static_headers: [] };
    const rows = [{ header: "DD-API-KEY", value: "rotated-secret" }];

    it("keeps saved discovery while no header rows are entered", () => {
      expect(getEditToolPreview({ ...savedCustom, secret_headers: [] }, savedCustom)).toEqual({ kind: "saved" });
    });

    it("previews entered rows as credentials.headers", () => {
      expect(getEditToolPreview({ ...savedCustom, secret_headers: rows }, savedCustom)).toEqual({
        kind: "preview",
        config: {
          url: "https://example.com/mcp",
          transport: "http",
          auth_type: "secret_headers",
          static_headers: {},
          credentials: { headers: { "DD-API-KEY": "rotated-secret" } },
        },
      });
    });

    it("needs rows before previewing a switch to secret headers", () => {
      expect(getEditToolPreview({ ...saved, auth_type: "secret_headers" }, saved).kind).toBe("incomplete");
      expect(getEditToolPreview({ ...saved, auth_type: "secret_headers", secret_headers: rows }, saved).kind).toBe(
        "preview",
      );
    });

    it("does not reuse the stored headers on a new origin", () => {
      expect(getEditToolPreview({ ...savedCustom, url: "https://other.example/mcp" }, savedCustom).kind).toBe(
        "incomplete",
      );
    });

    it("waits for a value on every row", () => {
      expect(
        getEditToolPreview({ ...savedCustom, secret_headers: [{ header: "DD-API-KEY", value: " " }] }, savedCustom)
          .kind,
      ).toBe("incomplete");
    });
  });

  it("waits while any secret header row is unnamed or has no value", () => {
    const edited = { ...saved, auth_type: "secret_headers", url: "https://example.com/corrected-mcp" };
    const complete = { header: "X-Api-Key", value: "first" };
    expect(getEditToolPreview({ ...edited, secret_headers: [complete] }, saved).kind).toBe("preview");
    for (const unfinished of [
      { header: "", value: "second" },
      { header: "X-Other", value: "" },
    ]) {
      expect(getEditToolPreview({ ...edited, secret_headers: [complete, unfinished] }, saved).kind).toBe("incomplete");
    }
  });
});
