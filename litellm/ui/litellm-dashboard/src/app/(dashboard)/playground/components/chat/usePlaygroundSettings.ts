import { useCallback, useEffect, useState } from "react";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";
import { defaultSettings, loadSettings, storedValues, type PlaygroundSettings } from "./playgroundSettings";

export type SettingsUpdate = Partial<PlaygroundSettings> | ((current: PlaygroundSettings) => PlaygroundSettings);

const browserStorage = {
  read: (key: string) => window.sessionStorage.getItem(key),
  readSecure: getSecureItem,
};

const persist = (settings: PlaygroundSettings): void => {
  storedValues(settings).forEach(({ key, value, secure }) => {
    try {
      if (value === null) window.sessionStorage.removeItem(key);
      else if (secure) setSecureItem(key, value);
      else window.sessionStorage.setItem(key, value);
    } catch {
      return;
    }
  });
};

export function usePlaygroundSettings(keysDisabled: boolean) {
  const [settings, setSettings] = useState<PlaygroundSettings>(() => loadSettings(browserStorage, keysDisabled));

  useEffect(() => persist(settings), [settings]);

  const update = useCallback(
    (change: SettingsUpdate) =>
      setSettings((current) => (typeof change === "function" ? change(current) : { ...current, ...change })),
    [],
  );

  const reset = useCallback(() => setSettings(defaultSettings(keysDisabled)), [keysDisabled]);

  return { settings, update, reset } as const;
}
