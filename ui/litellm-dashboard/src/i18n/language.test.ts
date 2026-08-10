import { describe, expect, it } from "vitest";
import { resolveLanguage } from "./language";

describe("resolveLanguage", () => {
  it.each([
    { saved: null, browser: "ru-RU", expected: "ru" },
    { saved: null, browser: "ru", expected: "ru" },
    { saved: null, browser: "en-US", expected: "en" },
    { saved: null, browser: "de-DE", expected: "en" },
    { saved: "en", browser: "ru-RU", expected: "en" },
    { saved: "ru", browser: "en-US", expected: "ru" },
    { saved: "de", browser: "ru-RU", expected: "ru" },
  ] as const)("returns $expected for saved=$saved and browser=$browser", ({ saved, browser, expected }) => {
    expect(resolveLanguage(saved, browser)).toBe(expected);
  });
});
