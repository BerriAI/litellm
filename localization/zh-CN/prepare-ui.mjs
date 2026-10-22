import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const root = path.resolve(process.argv[2]);
const packageRoot = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(root, "package.json"));
const ts = require("typescript");
const layoutPath = path.join(root, "src/app/layout.tsx");
const original = fs.readFileSync(layoutPath, "utf8");
const source = ts.createSourceFile(layoutPath, original, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const edits = [];
const fontImports = new Set();
const fontVariables = new Set();
const alreadyPrepared = original.includes('"@/lib/zh-cn-locale"');

for (const statement of source.statements) {
  if (ts.isImportDeclaration(statement) && statement.moduleSpecifier.text === "next/font/google") {
    for (const element of statement.importClause.namedBindings.elements) fontImports.add(element.name.text);
    edits.push({ start: statement.getFullStart(), end: statement.end, value: "" });
  }
}
for (const statement of source.statements) {
  if (!ts.isVariableStatement(statement)) continue;
  const declarations = statement.declarationList.declarations;
  const fonts = declarations.filter((declaration) => declaration.initializer
    && ts.isCallExpression(declaration.initializer) && fontImports.has(declaration.initializer.expression.getText(source)));
  if (fonts.length === 0) continue;
  if (fonts.length !== declarations.length) throw new Error("字体与其他变量共用声明，请拆开 layout.tsx 中的声明后再应用中文补丁");
  for (const declaration of fonts) fontVariables.add(declaration.name.getText(source));
  edits.push({ start: statement.getFullStart(), end: statement.end, value: "" });
}

let bodyCount = 0;
let htmlCount = 0;
function visit(node) {
  if (ts.isJsxOpeningElement(node) && node.tagName.getText(source) === "html") {
    htmlCount += 1;
    const attribute = node.attributes.properties.find((item) => ts.isJsxAttribute(item) && item.name.getText(source) === "lang");
    if (attribute !== undefined) edits.push({ start: attribute.initializer.getStart(source), end: attribute.initializer.end, value: '"zh-CN"' });
    else edits.push({ start: node.tagName.end, end: node.tagName.end, value: ' lang="zh-CN"' });
  }
  if (ts.isJsxOpeningElement(node) && node.tagName.getText(source) === "body") {
    bodyCount += 1;
    for (const attribute of node.attributes.properties) {
      if (!ts.isJsxAttribute(attribute) || attribute.name.getText(source) !== "className") continue;
      if (attribute.initializer === undefined || !ts.isJsxExpression(attribute.initializer)) continue;
      const expression = attribute.initializer.expression;
      if (expression === undefined || !ts.isPropertyAccessExpression(expression)) continue;
      if (fontVariables.has(expression.expression.getText(source))) {
        edits.push({ start: attribute.getStart(source), end: attribute.end, value: "" });
      }
    }
    if (!alreadyPrepared) edits.push({ start: node.end, end: node.end, value: "<ZhCnLocale>" });
  }
  if (ts.isJsxClosingElement(node) && node.tagName.getText(source) === "body" && !alreadyPrepared) {
    edits.push({ start: node.getStart(source), end: node.getStart(source), value: "</ZhCnLocale>" });
  }
  ts.forEachChild(node, visit);
}
visit(source);
if (htmlCount !== 1 || bodyCount !== 1) throw new Error("根布局结构已改变，请检查 src/app/layout.tsx 中的 html 和 body");
let result = original;
for (const edit of edits.sort((a, b) => b.start - a.start)) result = result.slice(0, edit.start) + edit.value + result.slice(edit.end);
if (!alreadyPrepared) result = 'import ZhCnLocale from "@/lib/zh-cn-locale";\n' + result;
fs.writeFileSync(layoutPath, result);
fs.copyFileSync(path.join(packageRoot, "zh-cn-locale.tsx"), path.join(root, "src/lib/zh-cn-locale.tsx"));
const cssPath = path.join(root, "src/app/globals.css");
const css = fs.readFileSync(cssPath, "utf8");
const chineseFont = 'html[lang="zh-CN"] body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }';
if (!css.includes(chineseFont)) fs.appendFileSync(cssPath, `\n${chineseFont}\n`);
console.log("已设置简体中文文档语言、日期格式和系统字体");
