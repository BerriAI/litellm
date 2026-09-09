import { describe, expect, it } from "vitest";

import { LOCALE_STORAGE_KEY } from "./localePreferences";
import {
  clearLocalePreference,
  readCookie,
  readStoredLocale,
  resolveLocale,
  writeLocalePreference,
  type LocaleEnv,
} from "./localePreferences";

interface MemoryState {
  storage: Map<string, string>;
  /** Simulated Set-Cookie directives applied in order. */
  cookieDirectives: string[];
  languages: string[];
  secure: boolean;
}

function makeEnv(state: MemoryState): LocaleEnv {
  const cookieString = () => {
    // Reconstruct a document.cookie string from the applied directives, keeping
    // the most recently set value per name.
    const values = new Map<string, string>();
    for (const directive of state.cookieDirectives) {
      const [nameAndValue] = directive.split(";");
      const idx = nameAndValue.indexOf("=");
      if (idx === -1) continue;
      const name = nameAndValue.slice(0, idx).trim();
      const value = nameAndValue.slice(idx + 1).trim();
      values.set(name, value);
    }
    return Array.from(values.entries())
      .map(([name, value]) => `${name}=${value}`)
      .join("; ");
  };

  return {
    getCookieString: cookieString,
    setCookie: (name, value, opts) => {
      state.cookieDirectives.push(
        `${name}=${encodeURIComponent(value)}; SameSite=Lax; path=/${opts.secure ? "; Secure" : ""}`,
      );
    },
    removeCookie: (name) => {
      state.cookieDirectives.push(`${name}=; Max-Age=0; SameSite=Lax; path=/`);
    },
    storageGet: (key) => state.storage.get(key) ?? null,
    storageSet: (key, value) => {
      state.storage.set(key, value);
    },
    storageRemove: (key) => {
      state.storage.delete(key);
    },
    browserLanguages: () => state.languages,
    isSecure: () => state.secure,
  };
}

function freshState(overrides: Partial<MemoryState> = {}): MemoryState {
  return {
    storage: new Map(),
    cookieDirectives: [],
    languages: ["en"],
    secure: false,
    ...overrides,
  };
}

describe("readCookie", () => {
  it("parses a simple cookie string", () => {
    expect(readCookie("a=1; b=2", "b")).toBe("2");
  });

  it("decodes URI-encoded values", () => {
    expect(readCookie("litellm.locale=zh-CN", LOCALE_STORAGE_KEY)).toBe("zh-CN");
  });

  it("returns null when the key is absent or the string is empty", () => {
    expect(readCookie("", "x")).toBeNull();
    expect(readCookie("a=1", "x")).toBeNull();
  });
});

describe("writeLocalePreference", () => {
  it("writes to both localStorage and cookie with SameSite=Lax and path=/", () => {
    const state = freshState();
    const env = makeEnv(state);

    writeLocalePreference("zh-CN", env);

    expect(state.storage.get(LOCALE_STORAGE_KEY)).toBe("zh-CN");
    const directive = state.cookieDirectives.find((d) => d.startsWith("litellm.locale="));
    expect(directive).toContain("SameSite=Lax");
    expect(directive).toContain("path=/");
    expect(directive).not.toContain("Secure");
  });

  it("attaches Secure when the environment reports it", () => {
    const state = freshState({ secure: true });
    const env = makeEnv(state);

    writeLocalePreference("zh-CN", env);

    const directive = state.cookieDirectives.find((d) => d.startsWith("litellm.locale="));
    expect(directive).toContain("Secure");
  });
});

describe("readStoredLocale", () => {
  it("reads from localStorage first", () => {
    const state = freshState({ storage: new Map([[LOCALE_STORAGE_KEY, "zh-CN"]]) });
    state.cookieDirectives.push(`${LOCALE_STORAGE_KEY}=en; path=/`);
    expect(readStoredLocale(makeEnv(state))).toBe("zh-CN");
  });

  it("falls back to the cookie when localStorage is empty", () => {
    const state = freshState();
    state.cookieDirectives.push(`litellm.locale=zh-CN; path=/`);
    expect(readStoredLocale(makeEnv(state))).toBe("zh-CN");
  });

  it("returns null when no preference exists", () => {
    expect(readStoredLocale(makeEnv(freshState()))).toBeNull();
  });

  it("treats an unsupported stored value as absent", () => {
    const state = freshState({ storage: new Map([[LOCALE_STORAGE_KEY, "fr"]]) });
    expect(readStoredLocale(makeEnv(state))).toBeNull();
  });
});

describe("clearLocalePreference", () => {
  it("removes the key from both layers", () => {
    const state = freshState({
      storage: new Map([[LOCALE_STORAGE_KEY, "zh-CN"]]),
      cookieDirectives: [`litellm.locale=zh-CN; path=/`],
    });
    const env = makeEnv(state);

    clearLocalePreference(env);

    expect(state.storage.has(LOCALE_STORAGE_KEY)).toBe(false);
    expect(state.cookieDirectives.at(-1)).toContain("Max-Age=0");
  });
});

describe("resolveLocale (D5 priority)", () => {
  it("prefers an explicit stored preference over browser language", () => {
    const state = freshState({
      storage: new Map([[LOCALE_STORAGE_KEY, "zh-CN"]]),
      languages: ["en"],
    });
    expect(resolveLocale(makeEnv(state))).toBe("zh-CN");
  });

  it("uses browser language when no stored preference exists", () => {
    expect(resolveLocale(makeEnv(freshState({ languages: ["zh-CN"] })))).toBe("zh-CN");
    expect(resolveLocale(makeEnv(freshState({ languages: ["zh", "en"] })))).toBe("zh-CN");
  });

  it("honours browser language order with en priority", () => {
    expect(resolveLocale(makeEnv(freshState({ languages: ["en", "zh-CN"] })))).toBe("en");
    expect(resolveLocale(makeEnv(freshState({ languages: ["zh-CN", "en"] })))).toBe("zh-CN");
  });

  it("falls back to en for unsupported browser languages", () => {
    expect(resolveLocale(makeEnv(freshState({ languages: ["fr", "de"] })))).toBe("en");
    expect(resolveLocale(makeEnv(freshState({ languages: [] })))).toBe("en");
  });
});
