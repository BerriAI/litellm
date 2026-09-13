import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/http/client";

const sonner = vi.hoisted(() => ({
  success: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
}));

vi.mock("sonner", () => ({ toast: sonner }));
vi.unmock("@/lib/toast");

import { toast } from "./toast";

const lastCall = (fn: ReturnType<typeof vi.fn>) => fn.mock.calls.at(-1) as [unknown, Record<string, unknown>];

describe("toast", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("plain kinds", () => {
    it.each([
      ["success", sonner.success, 4000],
      ["info", sonner.info, 4000],
      ["warning", sonner.warning, 6000],
      ["error", sonner.error, 6000],
    ] as const)("%s forwards the message with its default duration", (kind, fn, duration) => {
      toast[kind]("hello");
      expect(fn).toHaveBeenCalledWith("hello", { description: undefined, duration });
    });

    it("lets callers override duration and add a description", () => {
      toast.success("saved", { description: "Model x", durationMs: 1500 });
      expect(sonner.success).toHaveBeenCalledWith("saved", { description: "Model x", duration: 1500 });
    });

    it("dismiss clears every toast", () => {
      toast.dismiss();
      expect(sonner.dismiss).toHaveBeenCalledWith();
    });
  });

  describe("fromError title from the proxy error type", () => {
    it("reads the type out of an ApiError body envelope", () => {
      toast.fromError(
        new ApiError("Budget has been exceeded", 400, {
          error: { message: "Budget has been exceeded", type: "budget_exceeded", code: "400" },
        }),
      );
      expect(sonner.warning).toHaveBeenCalledWith("Budget Exceeded", {
        description: "Budget has been exceeded",
        duration: 6000,
      });
      expect(sonner.error).not.toHaveBeenCalled();
    });

    it("maps every *_access_denied type to Access Denied", () => {
      toast.fromError({ message: "no", type: "team_model_access_denied", code: "401" });
      expect(lastCall(sonner.error)[0]).toBe("Access Denied");
    });

    it("prefers a deliberate type over the HTTP status", () => {
      toast.fromError(new ApiError("expired", 400, { error: { message: "expired", type: "expired_key" } }));
      expect(lastCall(sonner.error)[0]).toBe("Authentication Error");
    });

    it.each(["auth_error", "internal_server_error"])(
      "ignores the proxy's catch-all type %s and trusts the status",
      (type) => {
        toast.fromError(
          new ApiError("Model with id=abc not found in db", 400, { error: { message: "x", type, code: "400" } }),
        );
        expect(lastCall(sonner.error)[0]).toBe("Request Error");
      },
    );

    it("reads type and code from a bare proxy payload object", () => {
      toast.fromError({ message: "Key with alias 'k' already exists.", type: "bad_request_error", code: "400" });
      expect(sonner.error).toHaveBeenCalledWith("Request Error", {
        description: "Key with alias 'k' already exists.",
        duration: 6000,
      });
    });

    it("parses a JSON envelope carried inside an Error message", () => {
      toast.fromError(
        new Error(JSON.stringify({ error: { message: "Team not found", type: "not_found_error", code: "404" } })),
      );
      expect(sonner.error).toHaveBeenCalledWith("Not Found", { description: "Team not found", duration: 6000 });
    });

    it("reads type and code from a JSON envelope embedded after a caller's prefix, keeping the prefix", () => {
      const envelope = JSON.stringify({
        error: { message: "Key with alias 'k' already exists.", type: "bad_request_error", code: "400" },
      });
      toast.fromError(`Error creating the key: Error: ${envelope}`);
      expect(sonner.error).toHaveBeenCalledWith("Request Error", {
        description: "Error creating the key: Error: Key with alias 'k' already exists.",
        duration: 6000,
      });
    });

    it("leaves a string alone when its braces are not a JSON envelope", () => {
      toast.fromError("Template {name} is invalid");
      expect(sonner.error).toHaveBeenCalledWith("Error", {
        description: "Template {name} is invalid",
        duration: 6000,
      });
    });
  });

  describe("fromError title from the HTTP status", () => {
    it.each([
      [400, "Request Error", sonner.error],
      [401, "Authentication Error", sonner.error],
      [403, "Access Denied", sonner.error],
      [404, "Not Found", sonner.error],
      [409, "Already Exists", sonner.error],
      [422, "Validation Error", sonner.error],
      [429, "Rate Limit Exceeded", sonner.warning],
      [418, "Request Error", sonner.error],
      [500, "Server Error", sonner.error],
      [503, "Service Unavailable", sonner.error],
      [502, "Server Error", sonner.error],
    ])("status %i becomes %s", (status, title, fn) => {
      toast.fromError(new ApiError("boom", status, "boom"));
      expect(fn).toHaveBeenCalledWith(title, { description: "boom", duration: 6000 });
    });

    it("reads an axios-style response status and nested data message", () => {
      toast.fromError({ response: { status: 403, data: { error: { message: "nope" } } } });
      expect(sonner.error).toHaveBeenCalledWith("Access Denied", { description: "nope", duration: 6000 });
    });

    it("reads a numeric status_code field", () => {
      toast.fromError({ status_code: 429, message: "slow down" });
      expect(sonner.warning).toHaveBeenCalledWith("Rate Limit Exceeded", { description: "slow down", duration: 6000 });
    });

    it("ignores non-HTTP code strings", () => {
      toast.fromError({ code: "ECONNREFUSED", message: "connection refused" });
      expect(sonner.error).toHaveBeenCalledWith("Error", { description: "connection refused", duration: 6000 });
    });
  });

  describe("fromError message extraction", () => {
    it("shows a raw string as an error with the generic title", () => {
      toast.fromError("Please select at least one model");
      expect(sonner.error).toHaveBeenCalledWith("Error", {
        description: "Please select at least one model",
        duration: 6000,
      });
    });

    it("unwraps the proxy's python-dict string form", () => {
      toast.fromError(new Error("{'error': 'Model not found'}"));
      expect(lastCall(sonner.error)[1].description).toBe("Model not found");
    });

    it("unwraps a JSON string envelope passed directly", () => {
      toast.fromError('{"error": {"message": "invalid json payload"}}');
      expect(lastCall(sonner.error)[1].description).toBe("invalid json payload");
    });

    it("joins FastAPI detail arrays", () => {
      toast.fromError({ detail: [{ msg: "field a required" }, { msg: "field b required" }] });
      expect(lastCall(sonner.error)[1].description).toBe("field a required; field b required");
    });

    it("lets a caller override the duration", () => {
      toast.fromError(new ApiError("x", 500, "x"), { durationMs: 10000 });
      expect(lastCall(sonner.error)[1].duration).toBe(10000);
    });
  });
});
