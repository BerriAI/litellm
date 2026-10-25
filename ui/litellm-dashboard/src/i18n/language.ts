import zhCN from "./zh-CN.json";
import extensionMessages from "./messages/extensions.zh-CN.json";
import logMessages from "./messages/logs.zh-CN.json";
import managementMessages from "./messages/management.zh-CN.json";
import modelMessages from "./messages/models.zh-CN.json";

// 各业务模块只合并一次，避免每次渲染重新复制整份词典
const dictionary: Readonly<Record<string, string>> = {
  ...zhCN,
  ...extensionMessages,
  ...logMessages,
  ...managementMessages,
  ...modelMessages,
};

export type Language = "en" | "zh-CN";
export type Translate = (message: string, values?: Readonly<Record<string, string | number>>) => string;

export const isLanguage = (value: unknown): value is Language => value === "en" || value === "zh-CN";

export const resolveLanguage = (preference: unknown, defaultLanguage: unknown): Language => {
  if (isLanguage(preference)) return preference;
  return isLanguage(defaultLanguage) ? defaultLanguage : "en";
};

// 英文原文同时作为回退文案，未覆盖页面不影响使用，不翻译业务数据
export const translate = (language: Language, message: string, values: Parameters<Translate>[1] = {}): string => {
  const template = language === "zh-CN" && Object.hasOwn(dictionary, message) ? dictionary[message] : message;
  return template.replace(/\{(\w+)\}/g, (placeholder, key: string) =>
    Object.hasOwn(values, key) ? String(values[key]) : placeholder,
  );
};

// 纯函数列配置默认保留英文，由页面传入当前语言的翻译函数
export const englishTranslate: Translate = (message, values) => translate("en", message, values);
