"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import React from "react";

import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";

const THEMES = [
  { value: "system", label: "System", Icon: Monitor },
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
] as const;

const ThemeToggle: React.FC = () => {
  const { theme, setTheme } = useTheme();

  return (
    <ButtonGroup role="radiogroup" aria-label="Theme">
      {THEMES.map(({ value, label, Icon }) => {
        const selected = theme === value;
        return (
          <Button
            key={value}
            role="radio"
            aria-checked={selected}
            aria-label={label}
            title={label}
            variant={selected ? "secondary" : "outline"}
            size="icon-xs"
            className={selected ? "text-foreground" : "text-muted-foreground"}
            onClick={() => setTheme(value)}
          >
            <Icon />
          </Button>
        );
      })}
    </ButtonGroup>
  );
};

export default ThemeToggle;
