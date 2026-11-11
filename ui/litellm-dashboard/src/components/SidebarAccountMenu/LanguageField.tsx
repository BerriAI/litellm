import { Select as SelectPrimitive } from "@base-ui/react/select";
import { CheckIcon } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";

import BetaBadge from "@/components/BetaBadge";
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import { Select, SelectContent, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  browserLanguageStore,
  DEFAULT_LANGUAGE,
  englishName,
  LanguageStore,
  Locales,
  LOCALES,
  nativeName,
  supportedLanguages,
} from "@/i18n/languages";

const TRANSLATIONS_URL = "https://github.com/BerriAI/litellm/tree/main/ui/litellm-dashboard/src/i18n/locales";
const CONTROL_ID = "account-menu-language";

type LanguageFieldProps = {
  locales?: Locales;
  store?: LanguageStore;
  reload?: () => void;
};

export default function LanguageField({
  locales = LOCALES,
  store = browserLanguageStore,
  reload = () => window.location.reload(),
}: LanguageFieldProps) {
  const { t, i18n } = useTranslation();
  const languages = supportedLanguages(locales);
  if (languages.length < 2) return null;

  const choose = (language: string | null) => {
    if (language === null || language === i18n.language) return;
    store.write(language);
    reload();
  };

  return (
    <Field className="gap-1.5 px-3 pt-2 pb-2.5">
      <FieldLabel htmlFor={CONTROL_ID} className="text-[13px] font-normal">
        <BetaBadge>{t("shell.sidebarAccountMenu.language.label")}</BetaBadge>
      </FieldLabel>
      <Select
        items={languages.map((language) => ({ value: language, label: nativeName(language) }))}
        value={i18n.language}
        onValueChange={choose}
      >
        <SelectTrigger id={CONTROL_ID} size="sm" className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent align="start" className="w-auto min-w-(--anchor-width)">
          {languages.map((language) => (
            <SelectPrimitive.Item
              key={language}
              value={language}
              className="flex w-full cursor-default items-center gap-1.5 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground"
            >
              <SelectPrimitive.ItemText className="whitespace-nowrap">{nativeName(language)}</SelectPrimitive.ItemText>
              <SelectPrimitive.ItemIndicator className="flex">
                <CheckIcon className="size-4" />
              </SelectPrimitive.ItemIndicator>
              {language !== DEFAULT_LANGUAGE && (
                <span className="ml-auto pl-6 whitespace-nowrap text-muted-foreground">{englishName(language)}</span>
              )}
            </SelectPrimitive.Item>
          ))}
        </SelectContent>
      </Select>
      <FieldDescription className="text-xs">
        <Trans
          i18nKey="shell.sidebarAccountMenu.language.fallbackNote"
          components={{
            githubLink: (
              <a href={TRANSLATIONS_URL} target="_blank" rel="noopener noreferrer" className="whitespace-nowrap" />
            ),
          }}
        />
      </FieldDescription>
    </Field>
  );
}
