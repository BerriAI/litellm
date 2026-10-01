#!/usr/bin/env node
// T-02 key 集合一致性校验 CLI（下方红色：只读、不写入字典）。
// 校验 en 与 zh-CN 同名 namespace 的 key 集合、嵌套结构、插值变量集合完全一致。
//
// 用法：
//   node scripts/i18n/check-keys.mjs
//   node scripts/i18n/check-keys.mjs --en <enDir> --zh <zhDir>
//
// 退出码：一致 -> 0；不一致 -> 1。适合 CI 门禁（G3）。

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { diffLocaleDirs, hasDiff } from "./check-keys-lib.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const DEFAULT_EN = resolve(root, "src/locales/en");
const DEFAULT_ZH = resolve(root, "src/locales/zh-CN");

function parseArgs(argv) {
  const args = { en: DEFAULT_EN, zh: DEFAULT_ZH };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--en" && argv[i + 1]) args.en = resolve(argv[++i]);
    else if (argv[i] === "--zh" && argv[i + 1]) args.zh = resolve(argv[++i]);
    else if (argv[i] === "--help" || argv[i] === "-h") args.help = true;
  }
  return args;
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(`T-02 key consistency checker

Usage:
  node scripts/i18n/check-keys.mjs [--en <enDir>] [--zh <zhDir>]

Default dirs (relative to dashboard root):
  --en  src/locales/en
  --zh  src/locales/zh-CN

Exit code 0 when key sets match, 1 when any difference is found.`);
  process.exit(0);
}

const { en, zh } = parseArgs(process.argv.slice(2));

let diff;
try {
  diff = diffLocaleDirs(en, zh);
} catch (e) {
  console.error(`check-keys: failed to read locale dirs (${e.message})`);
  console.error(`  en: ${en}`);
  console.error(`  zh: ${zh}`);
  process.exit(2);
}

const failed = hasDiff(diff);
let anyLine = false;

const print = (line) => {
  console.log(line);
  anyLine = true;
};

if (diff.enOnly.length > 0) {
  print(`[EN-ONLY] namespace 仅存在于 ${en}:`);
  for (const n of diff.enOnly) print(`  - ${n}`);
}
if (diff.zhOnly.length > 0) {
  print(`[ZH-ONLY] namespace 仅存在于 ${zh}:`);
  for (const n of diff.zhOnly) print(`  - ${n}`);
}

for (const ns of diff.namespaces) {
  const nsProblems =
    ns.onlyInEn ||
    ns.missing.length > 0 ||
    ns.extra.length > 0 ||
    ns.shapeMismatch.length > 0 ||
    ns.interpolationMismatch.length > 0;

  if (!nsProblems) {
    print(
      `[OK] ${ns.namespace}: key set matched (` +
        `${ns.extra.length} missing / ${ns.shapeMismatch.length} shape / ` +
        `${ns.interpolationMismatch.length} interpolation)`,
    );
    continue;
  }

  if (ns.onlyInEn) {
    print(`[DIFF] ${ns.namespace}: namespace 只有 ${en} 侧`);
    continue;
  }

  print(`[DIFF] ${ns.namespace}:`);
  if (ns.missing.length > 0) {
    print(`  missing (zh 有、en 无):`);
    for (const k of ns.missing) print(`    - ${k}`);
  }
  if (ns.extra.length > 0) {
    print(`  extra (en 有、zh 无):`);
    for (const k of ns.extra) print(`    - ${k}`);
  }
  if (ns.shapeMismatch.length > 0) {
    print(`  shape mismatch (叶子/对象不一致):`);
    for (const k of ns.shapeMismatch) print(`    - ${k}`);
  }
  if (ns.interpolationMismatch.length > 0) {
    print(`  interpolation mismatch ({{...}} 变量不一致):`);
    for (const { key: k, en: ev, zh: zv } of ns.interpolationMismatch)
      print(`    - ${k}: en=[${ev.join(", ")}] zh=[${zv.join(", ")}]`);
  }
}

if (!anyLine) {
  console.log(`OK: 未发现 ${en} / ${zh} 下的 locale 文件（或目录不存在）。`);
}

if (failed) {
  console.error("check-keys: FAILED — en/zh key sets are not consistent.");
  process.exit(1);
}
console.log("check-keys: PASS — en/zh key sets consistent.");
process.exit(0);
