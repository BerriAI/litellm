import buildAsyncValidatorUrlRegex from "@rc-component/async-validator/es/rule/url";
import { describe, expect, it } from "vitest";

import { ANTD_URL_REGEX, isAntdUrl, MAX_ANTD_URL_LENGTH } from "./antdUrl";

describe("isAntdUrl", () => {
  it("is compiled from the same pattern async-validator uses for rule type url", () => {
    const reference = buildAsyncValidatorUrlRegex();
    expect(ANTD_URL_REGEX.source).toBe(reference.source);
    expect(ANTD_URL_REGEX.flags).toBe(reference.flags);
  });

  it.each([
    "https://guard.example.com/v1/check",
    "http://localhost:4000",
    "www.example.com",
    "//example.com",
    "https://127.0.0.1:8080/path?q=1",
    "https://user:pass@example.com",
  ])("accepts %s the way antd does", (value) => {
    expect(isAntdUrl(value)).toBe(true);
  });

  it.each(["example.com", "", "not a url", "https://", "ftp:/example.com", "http://exa mple.com"])(
    "rejects %s the way antd does",
    (value) => {
      expect(isAntdUrl(value)).toBe(false);
    },
  );

  it("rejects a url longer than the 2048 characters antd allows", () => {
    const long = `https://example.com/${"a".repeat(MAX_ANTD_URL_LENGTH)}`;
    expect(ANTD_URL_REGEX.test(long)).toBe(true);
    expect(isAntdUrl(long)).toBe(false);
  });
});
