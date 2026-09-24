import { describe, it, expect } from "vitest";
import {
  isLikelyCopy,
  findJsxText,
  findCopyAttributes,
  findMessageEntries,
  scanContent,
} from "../../scripts/i18n/scan-hardcoded-lib.mjs";

describe("isLikelyCopy", () => {
  it("accepts user-facing phrases", () => {
    expect(isLikelyCopy("Save Changes")).toBe(true);
    expect(isLikelyCopy("Tenant not found")).toBe(true);
  });

  it("rejects numbers/symbols, identifiers, urls, paths, placeholders", () => {
    expect(isLikelyCopy("123")).toBe(false);
    expect(isLikelyCopy("42,000")).toBe(false);
    expect(isLikelyCopy("gpt-4o")).toBe(false); // 模型名
    expect(isLikelyCopy("api_keys")).toBe(false); // API 字段
    expect(isLikelyCopy("bg-red-500")).toBe(false); // CSS 类
    expect(isLikelyCopy("https://example.com")).toBe(false);
    expect(isLikelyCopy("/api/v1/models")).toBe(false);
    expect(isLikelyCopy("Hello {{name}}")).toBe(false); // 插值占位
    expect(isLikelyCopy("")).toBe(false);
  });
});

describe("findJsxText", () => {
  it("extracts visible JSX text between tags", () => {
    const hits = findJsxText("<Button>Save Changes</Button>");
    expect(hits.map((h) => h.text)).toEqual(["Save Changes"]);
  });
  it("ignores empty and expression containers", () => {
    expect(findJsxText("<div>{value}</div>").map((h) => h.text)).toEqual([]);
    expect(findJsxText("<div></div>").map((h) => h.text)).toEqual([]);
  });
});

describe("findCopyAttributes", () => {
  it("extracts a11y/copy attribute literals", () => {
    const hits = findCopyAttributes(`aria-label="Close dialog" title="More info"`);
    expect(hits.map((h) => [h.attr, h.text])).toEqual([
      ["aria-label", "Close dialog"],
      ["title", "More info"],
    ]);
  });
  it("does not match unrelated attributes", () => {
    expect(findCopyAttributes(`className="x" data-slot="y"`)).toEqual([]);
  });
});

describe("findMessageEntries", () => {
  it("finds toast/notify string literal args", () => {
    expect(findMessageEntries(`toast("Saved successfully")`).map((h) => [h.entry, h.text])).toEqual([
      ["toast", "Saved successfully"],
    ]);
  });
  it("skips non-string args", () => {
    expect(findMessageEntries(`toast(errorObj)`)).toEqual([]);
  });
});

describe("scanContent", () => {
  const SAMPLE = `
import { t } from "i18next";

export const label = "not-copy";           // identifier-style single token, excluded
export const aria = 'aria-label="Close dialog"';
function Comp() {
  return (
    <div>
      <button aria-label="Open settings">Open settings</button>
      <span>{t("common:already.translated")}</span>
      <span>Tenant unavailable</span>
      <div data-slot="wrapper">pure</div>
    </div>
  );
}
toast("Tenant created");
`;

  it("finds jsx-text, attributes, and message entries; skips t()/identifiers/data-slot", () => {
    const hits = scanContent(SAMPLE, "/fake/Comp.tsx");
    const texts = hits.map((h) => h.text);
    expect(texts).toContain("Open settings"); // 属性(attr) 与 JSX 文本
    expect(texts).toContain("Tenant unavailable"); // JSX 文本
    expect(texts).toContain("Tenant created"); // toast 入口
    // t() 已国际化、单 token 标识符、data-slot 内部纯词不应被命中
    expect(texts).not.toContain("common:already.translated");
    expect(texts).not.toContain("not-copy");
    expect(texts).not.toContain("pure");
    // 每个候选都带 file/line/kind
    for (const h of hits) {
      expect(h.file).toBe("/fake/Comp.tsx");
      expect(typeof h.line).toBe("number");
      expect(typeof h.kind).toBe("string");
    }
  });
});
