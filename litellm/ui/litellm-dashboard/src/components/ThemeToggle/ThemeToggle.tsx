"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import React from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";

const oppositeTheme = (isDark: boolean) => (isDark ? "light" : "dark");

const ThemeToggle: React.FC = () => {
  const { t } = useTranslation();
  const { setTheme, resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const label = isDark ? t("themeToggle.switchToLight") : t("themeToggle.switchToDark");

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      title={label}
      className="text-muted-foreground"
      onClick={() => setTheme(oppositeTheme(isDark))}
    >
      {isDark ? <Moon /> : <Sun />}
    </Button>
  );
};

export default ThemeToggle;
