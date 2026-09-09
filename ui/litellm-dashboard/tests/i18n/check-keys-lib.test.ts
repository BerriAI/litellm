import { describe, it, expect, afterAll } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  flattenDict,
  extractInterpolationVars,
  diffDicts,
  diffLocaleDirs,
  hasDiff,
} from "../../scripts/i18n/check-keys-lib.mjs";

describe("flattenDict", () => {
  it("flattens nested objects with dotted paths", () => {
    const flat = flattenDict({ a: { b: { c: "x" } }, d: "y" });
    expect(flat.get("a.b.c")).toEqual({ value: "x", isLeaf: true });
    expect(flat.get("d")).toEqual({ value: "y", isLeaf: true });
    // 分支节点也被记录形状
    expect(flat.get("a")).toEqual({ value: undefined, isLeaf: false });
  });
});

describe("extractInterpolationVars", () => {
  it("extracts {{var}} names", () => {
    expect([...extractInterpolationVars("Hello {{name}}, you have {{count}}")]).toEqual(["name", "count"]);
  });
  it("returns empty set for non-strings", () => {
    expect(extractInterpolationVars(42).size).toBe(0);
    expect(extractInterpolationVars("no placeholders").size).toBe(0);
  });
});

describe("diffDicts", () => {
  it("reports missing/extra/insert keys and interpolation mismatch", () => {
    const en = {
      title: "Hello",
      name: "Hi {{name}}",
      nested: { keep: "same", gone: "en only" },
    };
    const zh = {
      title: "你好",
      name: "你好 {{name}}，{{count}}", // 插值变量不一致（多 count）
      nested: { keep: "相同", added: "zh only" },
      brandNew: "新增",
    };
    const d = diffDicts(en, zh);
    expect(d.extra.sort()).toEqual(["nested.gone"]); // en 有、zh 无
    expect(d.missing.sort()).toEqual(["brandNew", "nested.added"]); // zh 有、en 无
    expect(d.shapeMismatch).toEqual([]);
    expect(d.interpolationMismatch.map((x) => x.key)).toEqual(["name"]);
  });

  it("reports shape mismatch when a leaf becomes an object", () => {
    const d = diffDicts({ key: "x" }, { key: { sub: "y" } });
    expect(d.shapeMismatch).toEqual(["key"]);
  });
});

describe("diffLocaleDirs", () => {
  const tmp = mkdtempSync(join(tmpdir(), "check-keys-test-"));
  const enDir = join(tmp, "en");
  const zhDir = join(tmp, "zh-CN");
  mkdirSync(enDir);
  mkdirSync(zhDir);

  const write = (dir: string, name: string, obj: unknown) =>
    writeFileSync(join(dir, `${name}.json`), JSON.stringify(obj));

  afterAll(() => rmSync(tmp, { recursive: true, force: true }));

  it("returns no diff when key sets match", () => {
    write(enDir, "common", { ok: "OK", cancel: "Cancel", greeting: "Hi {{name}}" });
    write(zhDir, "common", { ok: "确定", cancel: "取消", greeting: "你好 {{name}}" });
    const d = diffLocaleDirs(enDir, zhDir);
    expect(hasDiff(d)).toBe(false);
  });

  it("flags a namespace missing in zh (enOnly)", () => {
    // en 有 common，zh 没有
    write(enDir, "common", { ok: "OK" });
    write(enDir, "onlyEn", { x: "1" });
    write(zhDir, "common", { ok: "确定" });
    const d = diffLocaleDirs(enDir, zhDir);
    expect(d.enOnly).toContain("onlyEn");
    expect(hasDiff(d)).toBe(true);
  });

  it("flags interpolation mismatch across dirs", () => {
    write(enDir, "common", { greeting: "Hi {{name}}" });
    write(zhDir, "common", { greeting: "你好 {{name}}，{{count}}" });
    const d = diffLocaleDirs(enDir, zhDir);
    expect(d.namespaces.find((n) => n.namespace === "common").interpolationMismatch.length).toBe(1);
    expect(hasDiff(d)).toBe(true);
  });

  it("flags a namespace only present in zh (zhOnly)", () => {
    write(enDir, "common", { a: "1" });
    write(zhDir, "common", { a: "1" });
    write(zhDir, "zhOnlyNs", { b: "2" });
    const d = diffLocaleDirs(enDir, zhDir);
    expect(d.zhOnly).toContain("zhOnlyNs");
    expect(hasDiff(d)).toBe(true);
  });

  it("flags a missing key across dirs", () => {
    write(enDir, "common", { ok: "OK", cancel: "Cancel" });
    write(zhDir, "common", { ok: "确定" }); // 缺 cancel
    const d = diffLocaleDirs(enDir, zhDir);
    expect(d.namespaces.find((n) => n.namespace === "common").extra).toContain("cancel");
    expect(hasDiff(d)).toBe(true);
  });
});
