"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import React from "react";

import { Button } from "@/components/ui/button";
import { useLanguage } from "@/contexts/LanguageContext";

const ThemeToggle: React.FC = () => {
  const { t } = useLanguage();
  const { setTheme, resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const label = t(isDark ? "Switch to light mode" : "Switch to dark mode");

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      title={label}
      className="text-muted-foreground"
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      {isDark ? <Moon /> : <Sun />}
    </Button>
  );
};

export default ThemeToggle;
