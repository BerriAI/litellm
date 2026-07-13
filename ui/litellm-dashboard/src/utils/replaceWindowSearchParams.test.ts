import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { replaceWindowSearchParams } from "./replaceWindowSearchParams";

describe("replaceWindowSearchParams", () => {
  const originalLocation = window.location;
  const replaceState = vi.fn();

  beforeEach(() => {
    replaceState.mockClear();
    vi.stubGlobal("history", { ...window.history, replaceState });
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        pathname: "/ui/api-keys",
        search: "?virtual_key=abc&create=true",
        hash: "",
      },
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      value: originalLocation,
    });
    vi.unstubAllGlobals();
  });

  it("mutates search params and preserves pathname under /ui", () => {
    replaceWindowSearchParams((params) => {
      params.delete("virtual_key");
    });
    expect(replaceState).toHaveBeenCalledWith(null, "", "/ui/api-keys?create=true");
  });

  it("drops the query string entirely when empty", () => {
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        pathname: "/ui/api-keys",
        search: "?virtual_key=abc",
        hash: "",
      },
    });
    replaceWindowSearchParams((params) => {
      params.delete("virtual_key");
    });
    expect(replaceState).toHaveBeenCalledWith(null, "", "/ui/api-keys");
  });
});
