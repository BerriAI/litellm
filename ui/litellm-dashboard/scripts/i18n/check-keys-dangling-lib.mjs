// T-02b 悬空 key 校验库 —— 组件引用的 key ⊆ 已声明 key（TEST_TOOLS §2 组件引用校验）。
// 使用 TypeScript 编译器 AST（typescript 属既有依赖，不新增 package.json 项），
// 只读扫描 `.ts/.tsx` 中的 `t("ns:key")` 引用，对照 en locale 资源检查是否存在。
// 只报告，不改写，不自动生成 key。
//
// 保守策略（降低误报）：
//   - 仅当文件“i18n-active”（import 了 react-i18next 的 useTranslation，或 @/i18n 的 getI18n）
//     才执行引用提取，避免把无关的本地 `t()` 函数当成翻译调用。
//   - 仅匹配首参为字符串字面量的 `t(...)` 与 `xxx.t(...)` 调用；非字面量（模板/变量）跳过并
//     标注为“动态 key”，供人工复核。
//   - 无命名空间前缀的 key 视为 default namespace（defaultNS，来自 registry）。
//   - 复数分支：引用 `key` 时，声明侧可能出现 `key_one`/`key_other`/`key_0`，视为合法。

import { readFileSync, readdirSync, statSync } from "node:fs";
import { extname, join } from "node:path";
import ts from "typescript";
import { collectNamespaces } from "./check-keys-lib.mjs";

const defaultNS = "common";
const PLURAL_SUFFIXES = ["one", "other", "zero", "few", "many", "two"];
const IGNORED_DIRS = new Set(["node_modules", ".next", "out", "dist", ".git", "coverage"]);

// ---------- 文件收集 ----------
export function collectTsFiles(root) {
  const out = [];
  const walk = (dir) => {
    if (!statSync(dir, { throwIfNoEntry: false })?.isDirectory()) return;
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      const st = statSync(full);
      if (st.isDirectory()) {
        if (!IGNORED_DIRS.has(entry)) walk(full);
      } else if (st.isFile() && (extname(full) === ".ts" || extname(full) === ".tsx")) {
        // 排除类型声明文件 *.d.ts（extname 也是 .ts）
        if (!full.endsWith(".d.ts")) out.push(full);
      }
    }
  };
  const st = statSync(root, { throwIfNoEntry: false });
  if (!st) return out;
  if (st.isFile()) {
    if (extname(root) === ".ts" || extname(root) === ".tsx") out.push(root);
  } else {
    walk(root);
  }
  return out;
}

function normalizeImportSpecifier(spec) {
  return spec.replace(/^["']|["']$/g, "");
}

// 判断文件是否“i18n-active”：import useTranslation(@react-i18next) 或 import { getI18n } @/i18n。
export function isI18nActive(source) {
  const s = ts.createSourceFile("_", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let active = false;
  ts.forEachChild(s, (node) => {
    if (active) return;
    if (ts.isImportDeclaration(node) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      const mod = node.moduleSpecifier.text;
      const isReactI18next = mod === "react-i18next";
      const isLocalI18n = mod === "@/i18n" || mod.endsWith("/i18n") || mod.endsWith("/i18n/index");
      if (!isReactI18next && !isLocalI18n) return;
      const named = node.importClause?.namedBindings;
      if (named && ts.isNamedImports(named)) {
        for (const el of named.elements) {
          const name = el.name.text;
          if (name === "useTranslation" || name === "getI18n" || name === "I18nProvider") {
            active = true;
            return;
          }
        }
      }
    }
  });
  return active;
}

// 提取一个字符串字面量 k 的路径（支持点分隔），同时从声明字典查是否有 key 或复数后缀。
function keyExists(declFlat, path) {
  if (declFlat.has(path)) return true;
  // 复数/序数后缀：key_one / key_other / key_0 …（key 可能带子路径，仅最后一个段加后缀）
  for (const p of declFlat.keys()) {
    if (p.startsWith(path + "_")) {
      const suf = p.slice(path.length + 1);
      if (PLURAL_SUFFIXES.includes(suf) || /^\d+$/.test(suf)) return true;
    }
  }
  return false;
}

// 提取文件中所有翻译 key 引用，返回 { qualified, unqualified, dynamic } 的数组。
export function extractTranslationKeys(source, filePath) {
  const s = ts.createSourceFile(filePath, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const refs = [];
  let seenTIdentifier = false;
  const inTrans = new Set();

  // 记录 <Trans> 组件的 i18nKey 属性与 children 静态文本（策略性最小支持：仅 i18nKey）。
  function visit(node) {
    // Trans i18nKey="ns:key"
    if (ts.isJsxAttribute(node) && node.name.getText(s) === "i18nKey") {
      const init = node.initializer;
      if (init && ts.isStringLiteral(init)) {
        refs.push({ kind: "trans", key: init.text, line: s.getLineAndCharacterOfPosition(init.getStart(s)).line + 1 });
      }
    }
    // t("...") 或 obj.t("...")
    if (ts.isCallExpression(node)) {
      const expr = node.expression;
      let callee = null;
      if (ts.isPropertyAccessExpression(expr)) callee = expr.name.text;
      else if (ts.isIdentifier(expr)) callee = expr.text;
      if (callee === "t" && node.arguments.length > 0 && ts.isStringLiteral(node.arguments[0])) {
        seenTIdentifier = true;
        const key = node.arguments[0].text;
        const line = s.getLineAndCharacterOfPosition(node.getStart(s)).line + 1;
        if (key.includes(":")) refs.push({ kind: "qualified", key, line });
        else refs.push({ kind: "unqualified", key, line });
      } else if (
        callee === "t" &&
        node.arguments.length > 0 &&
        !ts.isStringLiteral(node.arguments[0]) &&
        !ts.isTemplateLiteral(node.arguments[0])
      ) {
        const line = s.getLineAndCharacterOfPosition(node.getStart(s)).line + 1;
        refs.push({ kind: "dynamic", key: node.arguments[0].getText(s), line });
      }
    }
  }

  function walkNode(node) {
    if (ts.isJsxSelfClosingElement(node) || ts.isJsxOpeningElement(node)) {
      const tag = node.tagName.getText(s);
      if (tag === "Trans") {
        inTrans.add(node);
      }
    }
    visit(node);
    ts.forEachChild(node, walkNode);
  }
  walkNode(s);

  // 过滤：callee 为 t 的调用，需文件 i18n-active 且（存在 useTranslation t 或 getI18n/Trans）。
  // seenTIdentifier 标记存在 t(...) 调用；i18n-active 在外层已判断。
  return { refs, active: seenTIdentifier };
}

/**
 * 主入口：扫描 root（目录/文件），对比 en locale 资源，返回悬空引用列表。
 * @param {string} root 目标目录或文件
 * @param {object} opts { enDir, localesDir }
 */
export function findDanglingReferences(root, { enDir, locale = "en" } = {}) {
  const enNamespaceFiles = collectNamespaces(enDir);
  const declFlatByNs = new Map();
  for (const [ns, obj] of enNamespaceFiles) {
    const flat = new Set();
    // 复用 flattenDict，取所有叶子路径
    // （手动扁平化，保留分支节点，但悬空校验只看叶子是否可达）
    (function walk(o, p) {
      for (const [k, v] of Object.entries(o)) {
        const path = p ? `${p}.${k}` : k;
        if (v !== null && typeof v === "object" && !Array.isArray(v)) walk(v, path);
        else flat.add(path);
      }
    })(obj, "");
    declFlatByNs.set(ns, flat);
  }
  const enNames = new Set(enNamespaceFiles.keys());

  const files = collectTsFiles(root);
  const dangling = [];

  for (const file of files) {
    const source = readFileSync(file, "utf8");
    if (!isI18nActive(source)) continue;
    const { refs } = extractTranslationKeys(source, file);
    const nsOf = (key) => {
      const idx = key.indexOf(":");
      if (idx === -1) return { ns: defaultNS, path: key };
      return { ns: key.slice(0, idx), path: key.slice(idx + 1) };
    };
    for (const ref of refs) {
      if (ref.kind === "dynamic") {
        dangling.push({ file, line: ref.line, kind: "dynamic", key: ref.key, ns: undefined });
        continue;
      }
      const { ns, path } = nsOf(ref.key);
      if (!enNames.has(ns)) {
        dangling.push({ file, line: ref.line, kind: "unknown-namespace", key: ref.key, ns });
        continue;
      }
      if (!keyExists(declFlatByNs.get(ns), path)) {
        dangling.push({ file, line: ref.line, kind: "missing-key", key: ref.key, ns });
      }
    }
  }
  return dangling;
}
