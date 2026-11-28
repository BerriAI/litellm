import { formatDateWithEnglishPattern } from "./formats";
import i18n from "./i18n";

const DATE = { year: "numeric", month: "short", day: "numeric" } as const;
const TIME = { hour: "2-digit", minute: "2-digit" } as const;
const TIMESTAMP = { ...DATE, ...TIME, second: "2-digit" } as const;

export const displayDate = (date: Date, language = i18n.language): string =>
  formatDateWithEnglishPattern(date, language, date.toLocaleDateString(), DATE);

export const displayTime = (date: Date, language = i18n.language): string =>
  formatDateWithEnglishPattern(date, language, date.toLocaleTimeString(undefined, TIME), TIME);

export const displayTimestamp = (date: Date, language = i18n.language, options?: Intl.DateTimeFormatOptions): string =>
  formatDateWithEnglishPattern(date, language, date.toLocaleString(undefined, options), options ?? TIMESTAMP);
