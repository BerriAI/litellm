#!/usr/bin/env node
// T-02b 悬空 key 校验 CLI —— 组件引用的 key ⊆ 已声明 key（只报告 + 可作为门禁退出码）。
// 用量：node scripts/i18n/check-keys-dangling.mjs [dir|file ...]
//   --en <dir>  指定 en locale 资源目录（默认 src/locales/en）
// 退出码：0 = 无缺 key；1 = 存在缺 key/未知 namespace/动态 key；2 = 用法/IO 错误。
//
// 说明：动态 key（首参非字面量）无法静态判定，作为人工复核提示列出，不计入失败（除非 --fail-on-dynamic）。

import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import { findDanglingReferences } from "./check-keys-dangling-lib.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const DEFAULT_EN = resolve(root, "src/locales/en");

function parseArgs(argv) {
  const args = { targets: [], en: DEFAULT_EN, failOnDynamic: false, help: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--en" && argv[i + 1]) args.en = resolve(argv[++i]);
    else if (a === "--fail-on-dynamic") args.failOnDynamic = true;
    else if (a === "--help" || a === "-h") args.help = true;
    else if (!a.startsWith("-")) args.targets.push(a);
  }
  return args;
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(`T-02b dangling translation-key checker

Usage:
  node scripts/i18n/check-keys-dangling.mjs [dir|file ...] [--en <enDir>] [--fail-on-dynamic]

Scans .ts/.tsx sources for t("ns:key") / <Trans i18nKey> references and verifies each
referenced key exists in the en locale resource. Files must import useTranslation /
getI18n to be considered i18n-active.

Exit code 0 when no missing keys; 1 when missing keys / unknown namespaces found
(or dynamic keys with --fail-on-dynamic); 2 on IO error.`);
  process.exit(0);
}

const { targets, en, failOnDynamic } = parseArgs(process.argv.slice(2));
if (targets.length === 0) {
  console.error("check-keys-dangling: no target dir/file given. Pass a path or --help.");
  process.exit(2);
}

let dangling;
try {
  dangling = findDanglingReferences(targets[0], { enDir: en });
  // 支持多个 target：合并结果（简单起见，逐 target 运行追加）
  for (const t of targets.slice(1)) {
    dangling = dangling.concat(findDanglingReferences(t, { enDir: en }));
  }
} catch (e) {
  console.error(`check-keys-dangling: failed (${e.message})`);
  process.exit(2);
}

const byKind = (k) => dangling.filter((d) => d.kind === k);
const missingKey = byKind("missing-key");
const unknownNs = byKind("unknown-namespace");
const dynamics = byKind("dynamic");

if (dangling.length === 0) {
  console.log(`check-keys-dangling: PASS — no dangling translation keys in ${targets.join(", ")}.`);
  process.exit(0);
}

if (missingKey.length > 0) {
  console.log(`[MISSING-KEY] ${missingKey.length} 引用的 key 未在 en locale 声明:`);
  for (const d of missingKey) console.log(`  ${d.file}:${d.line}  [${d.ns}] ${d.key}`);
}
if (unknownNs.length > 0) {
  console.log(`[UNKNOWN-NS] ${unknownNs.length} 引用了未注册的 namespace:`);
  for (const d of unknownNs) console.log(`  ${d.file}:${d.line}  ${d.key}`);
}
if (dynamics.length > 0) {
  console.log(`[DYNAMIC] ${dynamics.length} 引用为动态 key（无法静态校验，需人工复核）:`);
  for (const d of dynamics) console.log(`  ${d.file}:${d.line}  ${d.key}`);
}

const failed = missingKey.length > 0 || unknownNs.length > 0 || (failOnDynamic && dynamics.length > 0);
if (failed) {
  console.error("check-keys-dangling: FAILED — dangling/key issues found.");
  process.exit(1);
}
console.log("check-keys-dangling: WARN — 存在需人工复核的动态 key（未因动态 key 失败）。");
process.exit(0);
