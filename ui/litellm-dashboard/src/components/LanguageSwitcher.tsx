"use client";

import { Languages } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { isLanguage, LANGUAGE_STORAGE_KEY } from "@/i18n";
import { useTranslation } from "@/i18n/useTranslation";

export default function LanguageSwitcher() {
  const { t, i18n } = useTranslation();

  const selectLanguage = (language: string) => {
    if (!isLanguage(language)) return;
    void i18n.changeLanguage(language);
    try {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
    } catch {}
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={<Button variant="ghost" size="icon-sm" aria-label={t("Language")} title={t("Language")} />}
      >
        <Languages />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuRadioGroup value={i18n.language} onValueChange={selectLanguage}>
          <DropdownMenuRadioItem value="en">
            <span lang="en">English</span>
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="zh-CN">
            <span lang="zh-CN">简体中文</span>
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
