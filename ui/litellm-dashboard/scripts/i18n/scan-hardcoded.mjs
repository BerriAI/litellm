#!/usr/bin/env node
// T-03 硬编码字符串扫描 CLI —— 只报告，不自动改写，不自动生成 key（方案 §3.4）。
// 扫描 .ts / .tsx 中疑似"用户可见"的硬编码文案，输出供人工 review 的清单。
//
// 用法：
//   node scripts/i18n/scan-hardcoded.mjs [dir|file ...]
//   默认扫描 src/（若存在）。
//   可传多个目录/文件；目录会递归遍历 .ts/.tsx。
//
// 输出带 "REPORT ONLY" 标注；退出码 0（报告工具不因命中失败）。

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { scanContent } from "./scan-hardcoded-lib.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const DEFAULT_SRC = resolve(root, "src");

const IGNORED_DIRS = new Set(["node_modules", ".next", "out", "dist", ".git", "coverage"]);
const TARGET_EXT = new Set([".ts", ".tsx"]);

function collectFiles(path, acc = []) {
  const st = statSync(path);
  if (st.isFile()) {
    if (TARGET_EXT.has(extname(path))) acc.push(path);
    return acc;
  }
  for (const entry of readdirSync(path)) {
    const full = join(path, entry);
    if (statSync(full).isDirectory()) {
      if (IGNORED_DIRS.has(entry)) continue;
      collectFiles(full, acc);
    } else if (TARGET_EXT.has(extname(full))) {
      acc.push(full);
    }
  }
  return acc;
}

const inputs = process.argv.slice(2).filter((a) => !a.startsWith("-"));
const targets = inputs.length > 0 ? inputs : [DEFAULT_SRC];

const files = [];
for (const t of targets) {
  if (!existsSync(t)) {
    console.error(`scan-hardcoded: path not found: ${t}`);
    process.exit(2);
  }
  collectFiles(t, files);
}

// 按文件分组去重排序
const unique = [...new Set(files)].sort();
const grouped = new Map();
let total = 0;

for (const f of unique) {
  const source = readFileSync(f, "utf8");
  const hits = scanContent(source, f);
  if (hits.length > 0) grouped.set(f, hits);
  total += hits.length;
}

console.log(
  "TODO REPORT — 疑似硬编码\"用户可见\"文案候选，供人工 review，工具不做改写。",
);
console.log(`扫描 ${unique.length} 个文件，${total} 个候选命中。\n`);

for (const [f, hits] of grouped) {
  console.log(`# ${f}`);
  for (const h of hits) {
    console.log(`  ${h.line}:${h.col} [${h.kind}] ${JSON.stringify(h.text)}`);
  }
  console.log("");
}

if (total === 0) {
  console.log(`OK: 未发现疑似硬编码的用户可见文案候选。`);
}
