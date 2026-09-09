import { describe, it, expect, afterAll } from "vitest";
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  collectTsFiles,
  extractTranslationKeys,
  findDanglingReferences,
  isI18nActive,
} from "../../scripts/i18n/check-keys-dangling-lib.mjs";

function writeLocale(dir, name, obj) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, `${name}.json`), JSON.stringify(obj));
}

describe("collectTsFiles", () => {
  it("collects only .ts / .tsx and skips ignored dirs", () => {
    const tmp = mkdtempSync(join(tmpdir(), "dangling-collect-"));
    mkdirSync(join(tmp, "node_modules"), { recursive: true });
    mkdirSync(join(tmp, "src"), { recursive: true });
    writeFileSync(join(tmp, "src", "a.tsx"), "x");
    writeFileSync(join(tmp, "src", "b.ts"), "y");
    writeFileSync(join(tmp, "src", "c.d.ts"), "z");
    writeFileSync(join(tmp, "node_modules", "ignored.tsx"), "i");
    const files = collectTsFiles(tmp)
      .map((f) => f.split("/").slice(-2).join("/"))
      .sort();
    expect(files).toEqual(["src/a.tsx", "src/b.ts"]); // 排除 .d.ts 与 node_modules，且 a 在 b 前按序
    rmSync(tmp, { recursive: true, force: true });
  });
});

describe("isI18nActive", () => {
  it("true when importing useTranslation from react-i18next", () => {
    expect(isI18nActive(`import { useTranslation } from "react-i18next";`)).toBe(true);
  });
  it("true when importing getI18n from @/i18n", () => {
    expect(isI18nActive(`import { getI18n } from "@/i18n";`)).toBe(true);
  });
  it("false when no i18n import", () => {
    expect(isI18nActive(`import { useState } from "react"; function f(){ const t=(x)=>x; return t("hi"); }`)).toBe(
      false,
    );
  });
});

describe("extractTranslationKeys", () => {
  it("extracts qualified, unqualified, and <Trans i18nKey> refs", () => {
    const src = `
      import { useTranslation } from "react-i18next";
      export function C(){
        const { t } = useTranslation();
        return (<>
          <span>{t("common:hello")}</span>
          <span>{t("dashboard")}</span>
          <Trans i18nKey="navigation:dashboard">fallback</Trans>
        </>);
      }
    `;
    const { refs } = extractTranslationKeys(src, "/f/C.tsx");
    const kinds = refs.map((r) => r.kind).sort();
    expect(kinds).toContain("qualified");
    expect(kinds).toContain("unqualified");
    expect(refs.filter((r) => r.kind === "trans").map((r) => r.key)).toEqual(["navigation:dashboard"]);
  });

  it("marks non-string first args as dynamic", () => {
    const src = `
      import { useTranslation } from "react-i18next";
      export function C({k}){ const { t } = useTranslation(); return <span>{t(k)}</span>; }
    `;
    const { refs } = extractTranslationKeys(src, "/f/C.tsx");
    expect(refs.filter((r) => r.kind === "dynamic").length).toBe(1);
  });
});

describe("findDanglingReferences", () => {
  const tmp = mkdtempSync(join(tmpdir(), "dangling-"));
  const enDir = join(tmp, "en");
  const srcDir = join(tmp, "src");
  mkdirSync(srcDir, { recursive: true });

  writeLocale(enDir, "common", { hello: "Hello", action: { save: "Save" }, count: "{{count}} items" });
  writeLocale(enDir, "navigation", { dashboard: "Dashboard" });

  const writeSrc = (name, src) => writeFileSync(join(srcDir, name), src);

  afterAll(() => rmSync(tmp, { recursive: true, force: true }));

  it("flags a referenced key missing from the declared resource", () => {
    writeSrc(
      "Missing.tsx",
      `import { useTranslation } from "react-i18next";
       export function C(){ const { t } = useTranslation(); return <span>{t("common:hello")} {t("common:missing")}</span>; }`,
    );
    const d = findDanglingReferences(srcDir, { enDir });
    expect(d.some((x) => x.kind === "missing-key" && x.key === "common:missing")).toBe(true);
    // 已有 key 不应误报
    expect(d.some((x) => x.key === "common:hello")).toBe(false);
  });

  it("resolves unqualified keys against the default namespace", () => {
    writeSrc(
      "Default.tsx",
      `import { useTranslation } from "react-i18next";
       export function C(){ const { t } = useTranslation(); return <span>{t("hello")}</span>; }`,
    );
    const d = findDanglingReferences(srcDir, { enDir });
    expect(d.some((x) => x.key === "hello")).toBe(false);
  });

  it("accepts plural suffixes (count keys)", () => {
    writeSrc(
      "Plural.tsx",
      `import { useTranslation } from "react-i18next";
       export function C(){ const { t } = useTranslation(); return <span>{t("common:count", { count }) }</span>; }`,
    );
    const d = findDanglingReferences(srcDir, { enDir });
    // common.count 声明为 "{{count}} items"，无后缀也匹配（字面量 key 已存在）
    expect(d.some((x) => x.key === "common:count")).toBe(false);
  });

  it("flags unknown namespaces", () => {
    writeSrc(
      "UnknownNs.tsx",
      `import { useTranslation } from "react-i18next";
       export function C(){ const { t } = useTranslation(); return <span>{t("doesNotExist:someth")}</span>; }`,
    );
    const d = findDanglingReferences(srcDir, { enDir });
    expect(d.some((x) => x.kind === "unknown-namespace")).toBe(true);
  });
});
