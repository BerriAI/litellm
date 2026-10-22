import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { translateSource, preserveTerminology } from "../localize.mjs";

const require = createRequire(import.meta.url);
const ts = require(process.env.TYPESCRIPT_PATH);
const dictionary = {
  "API Key": "API 密钥", "Enter your API key": "请输入 API 密钥", "Save": "保存",
  "Active": "已启用", "Inactive": "未启用", "Delete": "删除", "Delete model": "删除模型",
  "Failed to save": "保存失败", "No models found": "未找到模型",
};

test("本地化可见表单文案，同时保留提交值、协议标识和状态判断", () => {
  const source = `
    const payload = { model: "model-v1", role: "user", content: "Save", api_key: "private" };
    export const view = (status: string) => <form>
      <label htmlFor="api_key">API Key</label>
      <input id="api_key" name="api_key" placeholder="Enter your API key" value="Save" />
      <button type="submit">Save</button>
      <span>{status === "active" ? "Active" : "Inactive"}</span>
    </form>;
  `;
  const result = translateSource(ts, "form.tsx", source, dictionary);
  assert.match(result.source, /API 密钥/);
  assert.match(result.source, /请输入 API 密钥/);
  assert.match(result.source, /\{"保存"\}/);
  assert.match(result.source, /status === "active" \? "已启用" : "未启用"/);
  assert.match(result.source, /name="api_key"/);
  assert.match(result.source, /value="Save"/);
  assert.match(result.source, /content: "Save"/);
  assert.match(result.source, /model: "model-v1"/);
});

test("动态文案翻译保留变量和模板表达式", () => {
  const source = 'toast.error(`Failed to save ${model.id}`);';
  const result = translateSource(ts, "toast.ts", source, dictionary);
  assert.equal(result.source, 'toast.error(`保存失败 ${model.id}`);');
});

test("选择项仅翻译标签，保留接口使用的 value 和对象属性名", () => {
  const source = 'const options = [{label: "Active", value: "active"}, {label: "Delete", value: "delete"}];';
  const result = translateSource(ts, "select.ts", source, dictionary);
  assert.match(result.source, /label: "已启用", value: "active"/);
  assert.match(result.source, /label: "删除", value: "delete"/);
});

test("上游新增文案明确列入报告，已应用中文补丁的文件可重复处理", () => {
  const first = translateSource(ts, "empty.tsx", '<div>No models found<button>New upstream action</button></div>', dictionary);
  assert.deepEqual(first.missing.map((item) => item.text), ["New upstream action"]);
  const second = translateSource(ts, "empty.tsx", first.source, dictionary);
  assert.equal(second.source, first.source);
  assert.equal(second.translated, 0);
});

test("代码示例、URL 和属性访问保持原样", () => {
  const source = 'const data = JSON.stringify({content: "Save"}); const code = <code>{"curl https://api.example.com/v1/models"}</code>; const value = titles["Save"];';
  assert.equal(translateSource(ts, "example.tsx", source, dictionary).source, source);
});

test("嵌入样式和代码片段原样保留，中文补丁可处理源文件中的内联样式", () => {
  const source = '<><style jsx>{`@keyframes loading { 0% { opacity: 0; } }`}</style><code>Save</code><span>Save</span></>';
  const result = translateSource(ts, "styles.tsx", source, dictionary);
  assert.match(result.source, /@keyframes loading \{ 0% \{ opacity: 0; \} \}/);
  assert.match(result.source, /<code>Save<\/code>/);
  assert.match(result.source, /<span>\{"保存"\}<\/span>/);
});

test("日期和数字采用中文地区设置，语言选项的接口取值保持原样", () => {
  const source = 'const date = value.toLocaleDateString("en-US"); const options = [{label:"English", value:"en-US"}];';
  const result = translateSource(ts, "date.ts", source, dictionary);
  assert.match(result.source, /toLocaleDateString\("zh-CN"\)/);
  assert.match(result.source, /value:"en-US"/);
});

test("包含引号和 JSON 示例的属性译文仍是有效 JSX", () => {
  const source = '<input placeholder="API Key" />';
  const result = translateSource(ts, "quote.tsx", source, {"API Key": '输入 "api_key"，例如 {"value":"key"}'});
  const parsed = ts.createSourceFile("quote.tsx", result.source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  assert.deepEqual(parsed.parseDiagnostics, []);
  assert.match(result.source, /placeholder=\{/);
});

test("表单 render prop 中的字符串属性解构保持协议名称", () => {
  const source = '<Field>{({ "aria-invalid": invalid, "aria-describedby": description }) => <input aria-invalid={invalid} aria-describedby={description} placeholder="API Key" />}</Field>';
  const result = translateSource(ts, "binding.tsx", source, { "aria-invalid": "无效", "aria-describedby": "说明", "API Key": "API 密钥" });
  assert.match(result.source, /"aria-invalid": invalid/);
  assert.match(result.source, /"aria-describedby": description/);
  assert.match(result.source, /API 密钥/);
});

test("A2A 消息类型和遥测状态位于嵌套对象中时仍保持接口取值", () => {
  const source = 'const rpcRequest = {params:{message:{kind:"message",parts:[{kind:"text",text:"Save"}]}}}; const trace = {summary:{status:"error"}};';
  const dictionary = {message:"消息",text:"文本",error:"错误",Save:"保存"};
  assert.equal(translateSource(ts, "request.ts", source, dictionary).source, source);
});

test("样式映射和函数参数保持原值，独立的标题映射可以翻译", () => {
  const source = 'const buttonVariants = {text:"ghost"}; const tooltip = (entity: "key" | "team" = "key") => entity; const STATUS_TITLES = {not_found:"No models found"};';
  const result = translateSource(ts, "mapping.ts", source, {ghost:"幽灵",key:"密钥","No models found":"未找到模型"});
  assert.match(result.source, /text:"ghost"/);
  assert.match(result.source, /= "key"/);
  assert.match(result.source, /not_found:"未找到模型"/);
});

test("只包含展示标签的联合类型与返回标签同步翻译", () => {
  const source = 'const getLabel = (): { readonly label: "Active" | "Inactive" } => ({label: "Active"});';
  const result = translateSource(ts, "label.ts", source, dictionary);
  assert.match(result.source, /label: "已启用" \| "未启用"/);
  assert.match(result.source, /label: "已启用"\}\)/);
});

test("属性类型内的 Pick 键名和索引访问保持类型系统原值", () => {
  const source = 'type Data = {summary: Pick<Stats,"count" | "average">; children: Props["children"]};';
  assert.equal(translateSource(ts, "types.ts", source, {count:"数量",average:"平均值",children:"子项"}).source, source);
});

test("AI 术语按用户要求保持英文，普通中文界面词保持中文", () => {
  assert.equal(preserveTerminology("Input Tokens and Prompt Embeddings", "输入令牌和提示词嵌入向量"), "输入Token和PromptEmbedding");
  assert.equal(preserveTerminology("Reasoning Guardrail", "推理防护规则"), "ReasoningGuardrail");
  assert.equal(preserveTerminology("Save changes", "保存更改"), "保存更改");
});

test("组件展示属性的默认值可以翻译，业务参数默认值保持原样", () => {
  const source = 'function Title({title="Save", entity="key"}) {return title;} function message(label="Save",entity="key"){return label;}';
  const result = translateSource(ts,"props.ts",source,{Save:"保存",key:"密钥"});
  assert.match(result.source,/title="保存", entity="key"/);
  assert.match(result.source,/label="保存",entity="key"/);
});

test("指标名称字典翻译展示值，业务索引变量保持原值", () => {
  const source = 'const METRIC_NOUN = {spend:"spend"}; const noun="investigation"; const state={metric:"spend"};';
  const result = translateSource(ts,"metric.ts",source,{spend:"费用",investigation:"调查"});
  assert.match(result.source,/spend:"费用"/);
  assert.match(result.source,/noun="investigation"/);
  assert.match(result.source,/metric:"spend"/);
});
