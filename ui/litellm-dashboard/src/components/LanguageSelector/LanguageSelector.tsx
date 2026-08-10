"use client";

import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useDashboardLanguage } from "@/i18n/I18nProvider";
import { resources } from "@/i18n/resources";
import { Languages } from "lucide-react";

const LanguageSelector = () => {
  const { language, setLanguage } = useDashboardLanguage();
  const copy = resources[language].common.language;
  const activeLanguageName = language === "ru" ? copy.russian : copy.english;

  return (
    <Popover>
      <PopoverTrigger
        className="inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
        aria-label={`${copy.selectorLabel}: ${activeLanguageName}`}
        title={copy.selectorTitle}
      >
        <Languages className="size-4" />
        <span>{language.toUpperCase()}</span>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-36 gap-1 p-1.5">
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start"
          aria-pressed={language === "en"}
          onClick={() => void setLanguage("en")}
        >
          {copy.english}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start"
          aria-pressed={language === "ru"}
          onClick={() => void setLanguage("ru")}
        >
          {copy.russian}
        </Button>
      </PopoverContent>
    </Popover>
  );
};

export default LanguageSelector;
