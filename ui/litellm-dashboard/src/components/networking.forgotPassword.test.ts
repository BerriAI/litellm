import { describe, it, expect, vi, beforeEach } from "vitest";
import { forgotPasswordCall, validateResetTokenCall, resetPasswordCall } from "./networking";

describe("forgot/reset password networking calls", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ message: "ok" }),
      }),
    );
  });

  it("forgotPasswordCall posts the email as JSON", async () => {
    await forgotPasswordCall("alice@example.com");
    const [url, options] = (fetch as any).mock.calls[0];
    expect(url).toContain("/user/forgot_password");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({ email: "alice@example.com" });
  });

  it("validateResetTokenCall issues a GET with the token as a query param", async () => {
    (fetch as any).mockResolvedValueOnce({ ok: true, json: async () => ({ user_email: "alice@example.com" }) });
    await validateResetTokenCall("tok-123");
    const [url, options] = (fetch as any).mock.calls[0];
    expect(url).toContain("/user/reset_password/validate?token=tok-123");
    expect(options.method).toBe("GET");
  });

  it("resetPasswordCall posts token and new_password as JSON", async () => {
    await resetPasswordCall("tok-123", "new-secret");
    const [url, options] = (fetch as any).mock.calls[0];
    expect(url).toContain("/user/reset_password");
    expect(JSON.parse(options.body)).toEqual({ token: "tok-123", new_password: "new-secret" });
  });

  it("throws with the derived error message on a non-ok response", async () => {
    (fetch as any).mockResolvedValueOnce({ ok: false, json: async () => ({ detail: { error: "boom" } }) });
    await expect(forgotPasswordCall("alice@example.com")).rejects.toThrow();
  });
});
