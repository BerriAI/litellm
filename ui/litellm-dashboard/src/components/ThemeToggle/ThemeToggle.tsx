"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import React from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const THEMES = [
  { value: "system", label: "System", Icon: Monitor, beta: false },
  { value: "light", label: "Light", Icon: Sun, beta: false },
  { value: "dark", label: "Dark", Icon: Moon, beta: true },
] as const;

const ThemeToggle: React.FC = () => {
  const { theme, setTheme, resolvedTheme } = useTheme();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <Button variant="ghost" size="icon-sm" aria-label="Theme" title="Theme" className="text-muted-foreground" />
        }
      >
        {resolvedTheme === "dark" ? <Moon /> : <Sun />}
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-40">
        <DropdownMenuRadioGroup value={theme ?? "light"} onValueChange={setTheme}>
          {THEMES.map(({ value, label, Icon, beta }) => (
            <DropdownMenuRadioItem key={value} value={value}>
              <Icon />
              {label}
              {beta && (
                <Badge
                  variant="secondary"
                  className="px-1 py-0 text-[10px] font-medium text-muted-foreground"
                  title="Dark mode is still being rolled out, so some surfaces may not be styled yet"
                >
                  Beta
                </Badge>
              )}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default ThemeToggle;
