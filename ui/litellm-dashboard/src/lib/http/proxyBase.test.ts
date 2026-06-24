import { describe, it, expect, beforeEach } from "vitest";
import {
  getProxyBaseUrl,
  switchToWorkerUrl,
  getGlobalLitellmHeaderName,
  setGlobalLitellmHeaderName,
} from "./proxyBase";

const WORKER_URL_KEY = "litellm_worker_url";

describe("proxyBase", () => {
  beforeEach(() => {
    window.localStorage.clear();
    switchToWorkerUrl(null);
    setGlobalLitellmHeaderName("Authorization");
  });

  describe("getProxyBaseUrl", () => {
    it("falls back to the window origin when no base is configured", () => {
      expect(getProxyBaseUrl()).toBe(window.location.origin);
    });
  });

  describe("switchToWorkerUrl", () => {
    it("points the base at a valid worker URL and persists it", () => {
      switchToWorkerUrl("https://worker.example.com");
      expect(getProxyBaseUrl()).toBe("https://worker.example.com");
      expect(window.localStorage.getItem(WORKER_URL_KEY)).toBe("https://worker.example.com");
    });

    it("rejects a non-http(s) scheme and leaves the base unchanged", () => {
      switchToWorkerUrl("javascript:alert(1)");
      expect(getProxyBaseUrl()).toBe(window.location.origin);
      expect(window.localStorage.getItem(WORKER_URL_KEY)).toBeNull();
    });

    it("clears the worker URL and falls back when passed null", () => {
      switchToWorkerUrl("https://worker.example.com");
      switchToWorkerUrl(null);
      expect(getProxyBaseUrl()).toBe(window.location.origin);
      expect(window.localStorage.getItem(WORKER_URL_KEY)).toBeNull();
    });
  });

  describe("global header name", () => {
    it("defaults to Authorization and reflects updates", () => {
      expect(getGlobalLitellmHeaderName()).toBe("Authorization");
      setGlobalLitellmHeaderName("x-litellm-api-key");
      expect(getGlobalLitellmHeaderName()).toBe("x-litellm-api-key");
    });
  });
});
