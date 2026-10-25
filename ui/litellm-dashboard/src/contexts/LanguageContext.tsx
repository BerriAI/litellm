"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { isLanguage, resolveLanguage, translate, type Language, type Translate } from "@/i18n/language";

interface LanguageContextValue {
  language: Language;
  preference: Language | null;
  setPreference: (language: Language | null) => void;
  storageUnavailable: boolean;
  t: Translate;
}

const englishContext: LanguageContextValue = {
  language: "en",
  preference: null,
  setPreference: () => undefined,
  storageUnavailable: false,
  t: (message, values) => translate("en", message, values),
};
const LanguageContext = createContext(englishContext);

const CHANGE_EVENT = "litellm-language-change";
const subscribe = (callback: () => void) => {
  window.addEventListener("storage", callback);
  window.addEventListener(CHANGE_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(CHANGE_EVENT, callback);
  };
};
const serverPreference = () => null;

export function LanguageProvider({
  children,
  defaultLanguage,
  userId,
  storageScope,
}: {
  children: ReactNode;
  defaultLanguage: unknown;
  userId: string | null;
  storageScope: string;
}) {
  // 按代理和账户隔离偏好，localStorage 只保存语言，不保存任何认证信息
  const identity = userId === null ? "guest" : `user:${encodeURIComponent(userId)}`;
  const storageKey = `litellm:ui-language:${encodeURIComponent(storageScope)}:${identity}`;
  const readPreference = useCallback(() => {
    try {
      const value = localStorage.getItem(storageKey);
      return isLanguage(value) ? value : null;
    } catch {
      return null;
    }
  }, [storageKey]);
  const storedPreference = useSyncExternalStore(subscribe, readPreference, serverPreference);
  const [temporary, setTemporary] = useState<{ key: string; value: Language | null } | null>(null);
  const storageUnavailable = temporary?.key === storageKey;
  const preference = storageUnavailable ? temporary.value : storedPreference;
  const language = resolveLanguage(preference, defaultLanguage);
  const t = useCallback<Translate>((message, values) => translate(language, message, values), [language]);

  useEffect(() => {
    const previous = document.documentElement.lang;
    document.documentElement.lang = language;
    return () => {
      document.documentElement.lang = previous;
    };
  }, [language]);

  const setPreference = useCallback(
    (value: Language | null) => {
      try {
        if (value === null) localStorage.removeItem(storageKey);
        else localStorage.setItem(storageKey, value);
        setTemporary(null);
        window.dispatchEvent(new Event(CHANGE_EVENT));
      } catch {
        console.warn("无法保存界面语言偏好，当前会话仍可切换语言");
        setTemporary({ key: storageKey, value });
      }
    },
    [storageKey],
  );

  return (
    <LanguageContext.Provider value={{ language, preference, setPreference, storageUnavailable, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export const useLanguage = () => useContext(LanguageContext);
