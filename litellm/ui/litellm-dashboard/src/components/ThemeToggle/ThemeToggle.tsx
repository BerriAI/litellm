"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import React from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";

const oppositeTheme = (isDark: boolean) => (isDark ? "light" : "dark");

const emptySubscribe = () => () => {};

export function useThemeToggle() {
  const { t } = useTranslation();
  const { setTheme, resolvedTheme } = useTheme();
  const mounted = React.useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );
  const isDark = mounted && resolvedTheme === "dark";
  const label = isDark ? t("themeToggle.switchToLight") : t("themeToggle.switchToDark");
  return { isDark, label, toggleTheme: () => setTheme(oppositeTheme(isDark)) };
}

const ThemeToggle: React.FC = () => {
  const { isDark, label, toggleTheme } = useThemeToggle();

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      title={label}
      className="text-muted-foreground"
      onClick={toggleTheme}
    >
      {isDark ? <Moon /> : <Sun />}
    </Button>
  );
};

export default ThemeToggle;
