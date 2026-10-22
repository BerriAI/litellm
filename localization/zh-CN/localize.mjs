import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const displayProperties = new Set([
  "label", "title", "description", "placeholder", "header", "heading", "subheading",
  "tooltip", "helpText", "helperText", "hint", "emptyText", "emptyMessage", "message",
  "errorMessage", "successMessage", "buttonText", "confirmText", "cancelText", "okText",
  "loadingText", "searchPlaceholder", "aria-label", "aria-description", "alt", "caption",
  "text", "children", "content", "summary", "subtitle", "noResultsText", "triggerLabel",
  "emptyLabel", "descriptionText", "tooltipText", "actionLabel", "loadingMessage",
  "groupLabel", "shortLabel", "displayName",
  "showForAdmin", "showForNonAdmin",
  "noDataMessage",
]);
const machineProperties = new Set([
  "value", "key", "id", "name", "type", "role", "href", "src", "className", "class",
  "htmlFor", "accessorKey", "accessor", "dataIndex", "field", "path", "endpoint", "url",
  "method", "model", "model_name", "api_base", "api_key", "variant", "size", "target",
  "rel", "testId", "data-testid", "data-slot", "align", "side", "format", "language",
]);
const messageMethods = new Set(["error", "success", "warning", "info", "loading", "message", "fromError"]);
const displayVariable = /(?:labels?|titles?|descriptions?|placeholders?|hints?|tooltips?|messages?|headings?|captions?|emptyText|buttonText|subtitles?|displayName|helpText|display)$/i;
const sourceExtensions = new Set([".tsx", ".ts", ".jsx", ".js"]);
const ignoredDirectories = new Set(["node_modules", ".next", "out", "__tests__", "__mocks__"]);
const terminology = JSON.parse(fs.readFileSync(new URL("./terminology.json", import.meta.url), "utf8"));

export function preserveTerminology(original, translation) {
  let result = translation;
  for (const term of terminology) {
    if (!new RegExp(term.source, "i").test(original)) continue;
    for (const wording of term.translations) result = result.replaceAll(wording, term.preferred);
  }
  return result;
}

export function normalize(text) {
  return text.replace(/\s+/g, " ").trim();
}

function propertyName(ts, node) {
  if (ts.isIdentifier(node) || ts.isStringLiteral(node)) return node.text;
  return "";
}

function isCode(text) {
  if (/[\u3400-\u9fff]/.test(text)) return true;
  if (!/[a-zA-Z]/.test(text)) return true;
  if (/^(?:https?:|data:|\/|\.|#|--)/.test(text)) return true;
  if (/^\$\{|^\^|^[A-Z][A-Z0-9_]*=|^[a-z][a-z0-9+.-]*:\S/.test(text)) return true;
  if (/^[a-zA-Z\d_$.-]+(?:_[a-zA-Z\d_$.-]+)+$/.test(text)) return true;
  if (/^[A-Za-z\d_.+-]*\d[A-Za-z\d_.+-]*$/.test(text)) return true;
  if (/^[a-zA-Z\d@_-]+\/[a-zA-Z\d@_/*.-]+$/.test(text)) return true;
  if (/^(?:curl\s|import\s|from\s|SELECT\s|const\s|export\s|npm\s|pip\s|docker\s|pnpm\s)/.test(text)) return true;
  if (/^@keyframes\b|(?:^|[\s'"])(?:client|resolver|response|result|request|config)\s*=/.test(text)) return true;
  if (/^\s*[\[{][\s\S]*[\]}]\s*$/.test(text)) return true;
  if (/^(?:[A-Za-z]+\.)+[A-Za-z]+$/.test(text)) return true;
  if (/^#[0-9a-fA-F]{3,8}$/.test(text)) return true;
  return false;
}

function isComparison(ts, node) {
  if (!ts.isBinaryExpression(node)) return false;
  return new Set([
    ts.SyntaxKind.EqualsEqualsToken, ts.SyntaxKind.EqualsEqualsEqualsToken,
    ts.SyntaxKind.ExclamationEqualsToken, ts.SyntaxKind.ExclamationEqualsEqualsToken,
    ts.SyntaxKind.InKeyword, ts.SyntaxKind.InstanceOfKeyword,
  ]).has(node.operatorToken.kind);
}

function isVisible(ts, node) {
  for (let owner = node.parent; owner; owner = owner.parent) {
    if (ts.isJsxElement(owner) && new Set(["code", "pre", "style", "script", "SyntaxHighlighter", "CodeBlock", "Code", "CodeEditor"]).has(owner.openingElement.tagName.getText())) return false;
  }
  if (ts.isJsxText(node)) return true;
  for (let owner = node.parent; owner; owner = owner.parent) {
    if (ts.isJsxElement(owner) || ts.isJsxSelfClosingElement(owner)) break;
    if (ts.isVariableDeclaration(owner) && ts.isIdentifier(owner.name)
      && /(?:payload|request|requestBody|requestData|body|response|Variants|Sizes|Classes|Colors|Styles|Attributes)$/i.test(owner.name.text)) return false;
    if (ts.isCallExpression(owner) && owner.expression.getText() === "JSON.stringify") return false;
  }
  let current = node;
  while (current.parent) {
    const parent = current.parent;
    if (ts.isParameter(parent)) return parent.initializer === current && ts.isIdentifier(parent.name)
      && displayProperties.has(parent.name.text);
    if (ts.isBindingElement(parent) && parent.propertyName === current) return false;
    if (ts.isBindingElement(parent) && parent.initializer === current) return ts.isIdentifier(parent.name)
      && displayProperties.has(parent.name.text);
    if (ts.isComputedPropertyName(parent)) return false;
    if (ts.isMethodDeclaration(parent) && parent.name === current) return false;
    if (ts.isTypeNode(parent)) {
      if (!ts.isLiteralTypeNode(parent)) return false;
      let owner = parent;
      while (owner.parent && ts.isUnionTypeNode(owner.parent)) owner = owner.parent;
      return owner.parent !== undefined && ts.isPropertySignature(owner.parent)
        && displayProperties.has(propertyName(ts, owner.parent.name));
    }
    if (ts.isImportDeclaration(parent) || ts.isExportDeclaration(parent)) return false;
    if (ts.isElementAccessExpression(parent) || ts.isCaseClause(parent) || isComparison(ts, parent)) return false;
    if (ts.isPropertyAssignment(parent)) {
      if (parent.name === current) return false;
      const name = propertyName(ts, parent.name);
      if (machineProperties.has(name)) return false;
      if (displayProperties.has(name)) return true;
      let owner = parent.parent;
      while (owner.parent && (ts.isAsExpression(owner.parent) || ts.isSatisfiesExpression(owner.parent) || ts.isParenthesizedExpression(owner.parent))) owner = owner.parent;
      return owner.parent !== undefined && ts.isVariableDeclaration(owner.parent)
        && ts.isIdentifier(owner.parent.name) && (displayVariable.test(owner.parent.name.text)
          || /(?:^|_)NOUNS?$/.test(owner.parent.name.text));
    }
    if (ts.isJsxAttribute(parent)) {
      const name = parent.name.getText();
      return displayProperties.has(name);
    }
    if (ts.isJsxExpression(parent)) {
      if (ts.isJsxAttribute(parent.parent)) {
        return displayProperties.has(parent.parent.name.getText());
      }
      return true;
    }
    if (ts.isVariableDeclaration(parent)) {
      return ts.isIdentifier(parent.name) && displayVariable.test(parent.name.text);
    }
    if (ts.isCallExpression(parent)) {
      const call = parent.expression;
      if (ts.isPropertyAccessExpression(call)) {
        const receiver = call.expression.getText();
        if (messageMethods.has(call.name.text) && /(?:toast|message|notification|notifications)/i.test(receiver)) return true;
        if (/(?:labelWithHint|toast|notify|alert|confirm|requiredRule)/i.test(call.name.text)) return true;
      }
      if (ts.isIdentifier(call) && /^(?:alert|confirm|prompt|requiredRule|labelWithHint)$/.test(call.text)) return true;
      return false;
    }
    if (ts.isNewExpression(parent)) {
      if (ts.isIdentifier(parent.expression) && /Error$/.test(parent.expression.text)) return true;
      current = parent;
      continue;
    }
    if (ts.isEnumMember(parent)) return false;
    if (ts.isReturnStatement(parent)) {
      let owner = parent.parent;
      while (owner && !ts.isFunctionLike(owner)) owner = owner.parent;
      if (owner && owner.name && (displayVariable.test(owner.name.getText())
        || /^(?:get|format|render).*(?:Text|Label|Title|Description|Message|DisplayName)$/.test(owner.name.getText()))) return true;
    }
    current = parent;
  }
  return false;
}

function collectFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (ignoredDirectories.has(entry.name)) return [];
      return collectFiles(file);
    }
    if (!sourceExtensions.has(path.extname(file))) return [];
    if (/\.(?:test|spec|test-d)\.[^.]+$/.test(file) || file.endsWith(".d.ts")) return [];
    return [file];
  });
}

export function inspectSource(ts, file, source) {
  const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, file.endsWith("x") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  if (sourceFile.parseDiagnostics.length > 0) {
    const diagnostic = sourceFile.parseDiagnostics[0];
    throw new Error(`${file}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, " ")}`);
  }
  const candidates = [];
  function visit(node) {
    const literal = ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)
      || ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node) || ts.isJsxText(node);
    if (literal && isVisible(ts, node)) {
      const text = normalize(node.text);
      if (!isCode(text)) {
        const location = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
        candidates.push({ text, node, line: location.line + 1 });
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  return { sourceFile, candidates };
}

function preserveSpacing(original, translated) {
  const leading = /^\s/.test(original) ? " " : "";
  const trailing = /\s$/.test(original) ? " " : "";
  return leading + translated + trailing;
}

export function translateSource(ts, file, source, dictionary) {
  const { sourceFile, candidates } = inspectSource(ts, file, source);
  const edits = [];
  const missing = [];
  let translated = 0;
  function localizeDate(node) {
    if (ts.isStringLiteral(node) && /^(?:en|en-US|en-GB)$/.test(node.text)) {
      const parent = node.parent;
      if ((ts.isCallExpression(parent) || ts.isNewExpression(parent)) && parent.arguments
        && parent.arguments[0] === node
        && /(?:toLocaleString|toLocaleDateString|toLocaleTimeString|Intl\.(?:NumberFormat|DateTimeFormat|RelativeTimeFormat|ListFormat|Collator))$/.test(parent.expression.getText(sourceFile))) {
        edits.push({ start: node.getStart(sourceFile), end: node.end, value: '"zh-CN"' });
      }
    }
    ts.forEachChild(node, localizeDate);
  }
  localizeDate(sourceFile);
  for (const { text, node, line } of candidates) {
    if (!Object.hasOwn(dictionary, text)) {
      missing.push({ text, line });
      continue;
    }
    const value = dictionary[text];
    if (typeof value !== "string" || value.trim().length === 0) throw new Error(`Invalid Chinese translation: ${text}`);
    const target = preserveTerminology(text, value);
    if (target === text) continue;
    const spaced = preserveSpacing(node.text, target);
    if (ts.isJsxText(node)) {
      edits.push({ start: node.pos, end: node.end, value: `{${JSON.stringify(spaced)}}` });
    } else if (ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) {
      const endOffset = ts.isTemplateTail(node) ? 1 : 2;
      const escaped = spaced.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
      edits.push({ start: node.getStart(sourceFile) + 1, end: node.end - endOffset, value: escaped });
    } else if (ts.isStringLiteral(node) && ts.isJsxAttribute(node.parent)) {
      edits.push({ start: node.getStart(sourceFile), end: node.end, value: `{${JSON.stringify(spaced)}}` });
    } else {
      edits.push({ start: node.getStart(sourceFile), end: node.end, value: JSON.stringify(spaced) });
    }
    translated += 1;
  }
  let result = source;
  for (const edit of edits.sort((left, right) => right.start - left.start)) {
    result = result.slice(0, edit.start) + edit.value + result.slice(edit.end);
  }
  return { source: result, translated, missing, candidates: candidates.length };
}

function argumentsFrom(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    if (!argv[index].startsWith("--") || argv[index + 1] === undefined) throw new Error("Options must use --name value");
    values.set(argv[index].slice(2), argv[index + 1]);
  }
  return values;
}

function required(values, name) {
  if (!values.has(name)) throw new Error(`Missing --${name}`);
  return path.resolve(values.get(name));
}

export function run(command, argv) {
  const args = argumentsFrom(argv);
  const root = required(args, "source");
  const require = createRequire(path.join(root, "package.json"));
  const ts = args.has("typescript") ? require(required(args, "typescript")) : require("typescript");
  const files = collectFiles(path.join(root, "src"));
  const dictionary = command === "apply" ? {
    ...JSON.parse(fs.readFileSync(required(args, "catalog"), "utf8")),
    ...JSON.parse(fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "overrides.json"), "utf8")),
  } : {};
  const contexts = JSON.parse(fs.readFileSync(new URL("./context-overrides.json", import.meta.url), "utf8"));
  const entries = new Map();
  const missing = new Map();
  let translated = 0;
  let affectedFiles = 0;
  for (const file of files) {
    const source = fs.readFileSync(file, "utf8");
    const relative = path.relative(root, file);
    const inspection = inspectSource(ts, file, source);
    for (const { text, line } of inspection.candidates) {
      if (!entries.has(text)) entries.set(text, { text, occurrences: 0, examples: [] });
      const entry = entries.get(text);
      entry.occurrences += 1;
      if (entry.examples.length < 3) entry.examples.push(`${relative}:${line}`);
    }
    if (command === "apply") {
      const fileDictionary = { ...dictionary };
      if (Object.hasOwn(contexts, relative)) Object.assign(fileDictionary, contexts[relative]);
      const result = translateSource(ts, file, source, fileDictionary);
      for (const item of result.missing) {
        if (!missing.has(item.text)) missing.set(item.text, []);
        if (missing.get(item.text).length < 3) missing.get(item.text).push(`${relative}:${item.line}`);
      }
      translated += result.translated;
      if (result.source !== source) {
        fs.writeFileSync(file, result.source);
        affectedFiles += 1;
      }
    }
  }
  const output = required(args, "output");
  if (command === "extract") {
    fs.writeFileSync(output, JSON.stringify([...entries.values()].sort((a, b) => b.occurrences - a.occurrences), null, 2) + "\n");
    console.log(`已提取 ${entries.size} 条文案，扫描 ${files.length} 个源文件`);
    return;
  }
  if (command !== "apply") throw new Error(`Unknown command: ${command}`);
  const report = {
    filesScanned: files.length, filesTranslated: affectedFiles, stringsTranslated: translated,
    uniqueMessages: entries.size, missing: [...missing].map(([text, examples]) => ({ text, examples })),
  };
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + "\n");
  console.log(`已翻译 ${translated} 处文案，涉及 ${affectedFiles} 个文件；新增待翻译文案 ${missing.size} 条，详见 ${output}`);
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === undefined) throw new Error("Usage: node localize.mjs extract|apply --source directory --output report.json");
  run(process.argv[2], process.argv.slice(3));
}
