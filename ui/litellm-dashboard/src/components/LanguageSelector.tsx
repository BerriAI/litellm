"use client";

import { useId } from "react";
import { useLanguage } from "@/contexts/LanguageContext";
import { isLanguage } from "@/i18n/language";

export default function LanguageSelector() {
  const id = useId();
  const { preference, setPreference, storageUnavailable, t } = useLanguage();

  return (
    <div className="space-y-1">
      <label className="sr-only" htmlFor={id}>
        {t("Interface language")}
      </label>
      {/* 原生选择器保留键盘操作与移动端语言名称，不依赖新组件库 */}
      <select
        id={id}
        value={preference ?? "default"}
        onChange={(event) => setPreference(isLanguage(event.target.value) ? event.target.value : null)}
        className="h-11 max-w-full rounded-md border border-input bg-background px-2 text-sm text-foreground focus-visible:outline-2 focus-visible:outline-ring md:h-8"
      >
        <option value="default">{t("Follow site default")}</option>
        <option value="en" lang="en">
          English
        </option>
        <option value="zh-CN" lang="zh-CN">
          简体中文
        </option>
      </select>
      {storageUnavailable && (
        <p role="status" className="max-w-64 text-xs text-muted-foreground">
          {t("Your language applies to this session only because browser storage is unavailable.")}
        </p>
      )}
    </div>
  );
}
