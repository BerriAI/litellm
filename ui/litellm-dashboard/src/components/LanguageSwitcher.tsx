"use client";

import { Check, Languages } from "lucide-react";
import { useTranslation } from "react-i18next";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { SupportedLanguage } from "@/i18n/i18n";
import { cn } from "@/lib/cva.config";

export default function LanguageSwitcher() {
  const { t, i18n } = useTranslation();
  const currentLanguage: SupportedLanguage = i18n.resolvedLanguage === "zh-CN" ? "zh-CN" : "en";
  const entries: ReadonlyArray<{ key: SupportedLanguage; label: string }> = [
    { key: "zh-CN", label: t("common.simplifiedChinese") },
    { key: "en", label: t("common.english") },
  ];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("common.language")}
        className={cn(buttonVariants({ variant: "ghost", size: "sm" }), "gap-1.5")}
      >
        <Languages />
        {currentLanguage === "zh-CN" ? "中文" : "EN"}
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-32">
        {entries.map((entry) => (
          <DropdownMenuItem key={entry.key} onClick={() => void i18n.changeLanguage(entry.key)}>
            <span>{entry.label}</span>
            {entry.key === currentLanguage && <Check className="ml-auto" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
