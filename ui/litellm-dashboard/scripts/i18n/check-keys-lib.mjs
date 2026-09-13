// T-02 key 集合一致性校验库（纯 Node，无第三方依赖）。
// 对比 en / zh-CN 同名 namespace 的：
//   1. 递归扁平化 key 集合（含嵌套路径，点分隔）
//   2. 嵌套结构形状（叶子 vs 对象）
//   3. 每叶子上的 {{var}} 插值变量集合
// 输出对称差清单，供 CLI / CI 门禁（G3）消费。
// 只读、不改写任何文件。

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const DEFAULT_LOCALES = ["en", "zh-CN"];

// 从 JSON 值中提取插值变量 `{{name}}`（i18next 标准插值）。
export function extractInterpolationVars(value) {
  const vars = new Set();
  if (typeof value !== "string") return vars;
  const re = /\{\{\s*([A-Za-z0-9_][A-Za-z0-9_.]*)\s*\}\}/g;
  let m;
  while ((m = re.exec(value)) !== null) {
    vars.add(m[1]);
  }
  return vars;
}

// 递归扁平化一个字典：返回 Map<keyPath, { value, isLeaf }>。
// 对象值递归，叶子（字符串/数字/布尔/null/数组）记录 value。
export function flattenDict(obj, prefix = "") {
  const out = new Map();
  for (const [k, v] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      const sub = flattenDict(v, path);
      for (const [p, info] of sub) out.set(p, info);
      out.set(path, { value: undefined, isLeaf: false }); // 记录分支节点形状
    } else {
      out.set(path, { value: v, isLeaf: true });
    }
  }
  return out;
}

function parseJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

// 列出目录下顶层文件（不含子目录），去扩展名，返回 { name, absPath, ext }。
export function listLocaleFiles(dir) {
  if (!statSync(dir, { throwIfNoEntry: false })?.isDirectory()) return [];
  return readdirSync(dir)
    .filter((f) => statSync(join(dir, f)).isFile())
    .map((f) => {
      const dot = f.lastIndexOf(".");
      const name = dot > 0 ? f.slice(0, dot) : f;
      return { name, absPath: join(dir, f), ext: dot > 0 ? f.slice(dot + 1) : "" };
    });
}

// 命名空间之间没有共同文件也能校验；只对有同名校的文件做比对。
export function collectNamespaces(localeDir) {
  const files = listLocaleFiles(localeDir).filter((f) => f.ext === "json");
  return new Map(files.map((f) => [f.name, parseJson(f.absPath)]));
}

// 对比一对字典（en vs zh），返回差异详情。
export function diffDicts(a, b) {
  const fa = flattenDict(a);
  const fb = flattenDict(b);
  const aPaths = new Set(fa.keys());
  const bPaths = new Set(fb.keys());

  const missing = []; // 在 b 中有、a 中没有（即 zh 缺 en 的 key）
  const extra = []; // 在 a 中有、b 没有
  const shapeMismatch = [];
  const interpolationMismatch = [];

  // 以 a (en) 为基准：遍历所有叶子路径
  for (const p of aPaths) {
    if (!bPaths.has(p)) {
      extra.push(p); // en 有、zh 无 -> zh 缺
    }
  }
  for (const p of bPaths) {
    if (!aPaths.has(p)) {
      missing.push(p); // zh 有、en 无
    }
  }

  // 结构/插值比对仅在双方都存在叶子时进行
  for (const p of aPaths) {
    if (!bPaths.has(p)) continue;
    const infoA = fa.get(p);
    const infoB = fb.get(p);
    if (infoA.isLeaf !== infoB.isLeaf) {
      shapeMismatch.push(p);
      continue;
    }
    if (!infoA.isLeaf) continue;
    const va = extractInterpolationVars(infoA.value);
    const vb = extractInterpolationVars(infoB.value);
    const vaArr = [...va].sort();
    const vbArr = [...vb].sort();
    if (vaArr.join("|") !== vbArr.join("|")) {
      interpolationMismatch.push({ key: p, en: vaArr, zh: vbArr });
    }
  }

  return { missing, extra, shapeMismatch, interpolationMismatch };
}

// 顶层入口：给定 en 目录与 zh 目录，返回每个共同 namespace 的差异。
// 同时报告只有一方存在的 namespace（enOnly / zhOnly）。
export function diffLocaleDirs(enDir, zhDir, { locales = DEFAULT_LOCALES } = {}) {
  const enNs = collectNamespaces(enDir);
  const zhNs = collectNamespaces(zhDir);
  const enNames = new Set(enNs.keys());
  const zhNames = new Set(zhNs.keys());

  const results = [];
  for (const name of enNames) {
    if (zhNames.has(name)) {
      const d = diffDicts(enNs.get(name), zhNs.get(name));
      results.push({ namespace: name, ...d });
    } else {
      results.push({
        namespace: name,
        missing: [],
        extra: [],
        shapeMismatch: [],
        interpolationMismatch: [],
        onlyInEn: true,
      });
    }
  }
  const onlyInZh = [...zhNames].filter((n) => !enNames.has(n));
  return { namespaces: results, enOnly: [...enNames].filter((n) => !zhNames.has(n)), zhOnly: onlyInZh };
}

export function hasDiff(diff) {
  if (diff.enOnly.length > 0 || diff.zhOnly.length > 0) return true;
  return diff.namespaces.some(
    (n) =>
      n.missing.length > 0 ||
      n.extra.length > 0 ||
      n.shapeMismatch.length > 0 ||
      n.interpolationMismatch.length > 0 ||
      n.onlyInEn,
  );
}
