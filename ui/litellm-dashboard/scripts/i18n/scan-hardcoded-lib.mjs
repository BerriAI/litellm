// T-03 硬编码字符串扫描库（纯 Node，无第三方依赖）。
// 只报告疑似"用户可见"的硬编码文案，绝不改写代码、绝不自动生成 key。
// 供人工 review（方案 §3.4 硬约束：工具只报告）。
//
// 策略（line-based 启发式，适合作为报告工具而非精确解析器）：
//   1. JSX 文本：`>文案<` 形态的非空白/非注释/非纯符号文本。
//   2. 可访问性/文案属性字面量：aria-label / title / placeholder / alt / label。
//   3. 常见未国际化消息入口：toast( ... ) / notify( ... ) / useMessage / useToast。
// 排除：
//   - 已 t() / <Trans> 包裹的文案
//   - 纯数字/符号/URL/路径/正则/占位符
//   - 标识符（模式串）、CSS 类、data-slot、样式令牌
//   - 明显是模型名/API 字段/日志/代码示例的标识符

import { readFileSync } from "node:fs";

// ---------- 单个候选 ----------
// { file, line, col, kind, text }

// 判文案属性值是否"疑似用户可见文案"（过滤纯符号/数字/占位）。
const PLACEHOLDER_RE = /[{}]/;
const URL_RE = /^(https?:)?\/\//i;
const PATH_RE = /^[./~][\w./-]*$/;

export function isLikelyCopy(s) {
  const t = s.trim();
  if (t.length === 0) return false;
  // 纯数字/符号组合（不含字母）
  if (/^[\d\s.,%:+\-*/()_'"`]+$/.test(t)) return false;
  // 占位符（含 {{ }} 或 { } 渲染表达式）——由插值负责，不视为硬编码
  if (PLACEHOLDER_RE.test(t)) return false;
  // URL / 路径
  if (URL_RE.test(t) || PATH_RE.test(t)) return false;
  // 单 token 且形式为 CSS 类/标识符（连字符/下划线/无空格）——排除模型名、类名、变量
  if (/^[A-Za-z0-9][\w-]*$/.test(t)) return false;
  // 必须含字母
  if (!/[A-Za-z]/.test(t)) return false;
  // 至少两个词，保证是短语/句子而非单词级标识符片段
  const words = t.split(/\s+/).filter(Boolean);
  if (words.length < 2) return false;
  return true;
}

// 过滤已 <Trans> 包裹（或其参数为 t() 表达式）的文本。
// 行内出现 <Trans 时，本行 JSX 文本很可能已国际化，跳过以避免误报；
// 其余候选（属性/入口）由各自模式精确排除表达式与单 token，无需整行 t() 判断。
export function lineHasTransWrapper(context) {
  return /<Trans[\s>]/.test(context);
}

// 从单行提取 `attr="value"` 形态的属性字面量。
// attr 限定于文案属性集合。
const ATTR_TEXT_RE = /\b(aria-label|title|placeholder|alt)="([^"]*)"/g;
export function findCopyAttributes(line) {
  const out = [];
  let m;
  ATTR_TEXT_RE.lastIndex = 0;
  while ((m = ATTR_TEXT_RE.exec(line)) !== null) {
    out.push({ attr: m[1], text: m[2], col: m.index });
  }
  return out;
}

// 从 JSX 文本提取 `>text<`。只捕获紧跟 `>`、后跟 `</` 或 `/>` 的文本节点。
// 形如：<Tag>hello world</Tag> 或 <Tag>hi</Tag>
export function findJsxText(line) {
  const out = [];
  const re = />([^<>{}\n]+)</g;
  let m;
  while ((m = re.exec(line)) !== null) {
    const t = m[1].trim();
    if (t.length > 0) {
      out.push({ text: t, col: m.index + 1 });
    }
  }
  return out;
}

// 常见未国际化消息入口：toast( / notify( / setToast( / useMessage / enqueueSnackbar / addToast
const MSG_ENTRY_RE = /\b(toast|notify|enqueueSnackbar|addToast|showToast|useMessage|useToast)\s*\(([^)]*)\)/g;
export function findMessageEntries(line) {
  const out = [];
  let m;
  MSG_ENTRY_RE.lastIndex = 0;
  while ((m = MSG_ENTRY_RE.exec(line)) !== null) {
    const arg = m[2].trim();
    // 仅字符串字面量（单/双引号）视为候选；对象/表达式跳过
    const strHit = arg.match(/^(['"])(.*?)\1/);
    if (strHit) {
      out.push({ entry: m[1], text: strHit[2], col: m.index });
    }
  }
  return out;
}

// 主扫描：给定文件绝对路径与内容行数组，返回候选列表。
export function scanContent(source, filePath) {
  const lines = source.split("\n");
  const candidates = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const stripped = line.replace(/\/\/.*$/, "").replace(/\/\*[\s\S]*?\*\//g, "");
    if (stripped.trim() === "") continue;

    // 该行是否含 <Trans>（其 JSX 文本已国际化，跳过 JSX 文本候选项）
    const trans = lineHasTransWrapper(line);

    // 1) 文案属性（aria-label/title/placeholder/alt 的纯字符串字面量；
    //    t("...") 形式是 aria-label={t(...)}，不匹配本引号模式，天然排除）
    for (const { attr, text, col } of findCopyAttributes(stripped)) {
      if (isLikelyCopy(text)) {
        candidates.push({ file: filePath, line: i + 1, col, kind: `attr:${attr}`, text });
      }
    }

    // 2) JSX 文本
    if (!trans) {
      for (const { text, col } of findJsxText(stripped)) {
        if (isLikelyCopy(text)) {
          candidates.push({ file: filePath, line: i + 1, col, kind: "jsx-text", text });
        }
      }
    }

    // 3) 消息入口
    for (const { entry, text, col } of findMessageEntries(line)) {
      if (isLikelyCopy(text)) {
        candidates.push({ file: filePath, line: i + 1, col, kind: `entry:${entry}`, text });
      }
    }
  }
  return candidates;
}

export function scanFile(filePath) {
  const source = readFileSync(filePath, "utf8");
  return scanContent(source, filePath);
}
