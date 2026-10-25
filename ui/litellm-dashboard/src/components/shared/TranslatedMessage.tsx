"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import type { Translate } from "@/i18n/language";

// 仅用于调用方明确标记的固定界面文案，随实际渲染位置的语言上下文更新。
export default function TranslatedMessage({ message, values }: { message: string; values?: Parameters<Translate>[1] }) {
  const { t } = useLanguage();
  return <>{t(message, values)}</>;
}
