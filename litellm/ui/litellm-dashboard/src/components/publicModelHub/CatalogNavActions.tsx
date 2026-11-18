"use client";

import { ArrowUpRight, Ellipsis, Moon, Search, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { DOCS_URL } from "@/components/Navbar/DocsLink/DocsLink";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import styles from "./catalog.module.css";

export function CatalogNavActions({
  search,
  onSearchChange,
}: {
  search: string;
  onSearchChange: (value: string) => void;
}) {
  const { t } = useTranslation();
  const { resolvedTheme, setTheme } = useTheme();
  const hideBlog = useDisableBlogPosts();
  const hideCommunity = useDisableShowPrompts();
  const inputRef = useRef<HTMLInputElement>(null);
  const isDark = resolvedTheme === "dark";
  const themeLabel = t(isDark ? "themeToggle.switchToLight" : "themeToggle.switchToDark");

  useEffect(() => {
    const focusSearch = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== "k" || !(event.metaKey || event.ctrlKey)) return;
      event.preventDefault();
      inputRef.current?.focus();
      inputRef.current?.select();
    };
    window.addEventListener("keydown", focusSearch);
    return () => window.removeEventListener("keydown", focusSearch);
  }, []);

  return (
    <>
      <div className={styles.navSearch}>
        <Search size={14} aria-hidden="true" />
        <input
          ref={inputRef}
          type="search"
          aria-label={t("publicModelHub.catalog.searchModels")}
          placeholder={t("publicModelHub.catalog.searchModels")}
          value={search}
          onChange={(event) => onSearchChange(event.target.value)}
          aria-keyshortcuts="Meta+K Control+K"
        />
        <kbd aria-hidden="true">⌘ K</kbd>
      </div>
      <div className={styles.navActions}>
        <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className={styles.navDocs}>
          {t("shell.docsLink.label")}
          <ArrowUpRight size={14} strokeWidth={1} aria-hidden="true" />
        </a>
        <button
          type="button"
          className={styles.navIconButton}
          aria-label={themeLabel}
          title={themeLabel}
          onClick={() => setTheme(isDark ? "light" : "dark")}
        >
          <Moon size={18} className="dark:hidden" aria-hidden="true" />
          <Sun size={18} className="hidden dark:block" aria-hidden="true" />
        </button>
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <button
                type="button"
                className={styles.navIconButton}
                aria-label={t("publicModelHub.catalog.moreNavigation")}
              />
            }
          >
            <Ellipsis size={20} aria-hidden="true" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" sideOffset={11} className={`${styles.typography} ${styles.navMenu}`}>
            <DropdownMenuItem render={<a href={DOCS_URL} target="_blank" rel="noopener noreferrer" />}>
              {t("shell.docsLink.label")}
            </DropdownMenuItem>
            {!hideBlog && (
              <DropdownMenuItem
                render={<a href="https://docs.litellm.ai/blog" target="_blank" rel="noopener noreferrer" />}
              >
                {t("shell.blogDropdown.trigger")}
              </DropdownMenuItem>
            )}
            {!hideCommunity && (
              <>
                <DropdownMenuItem
                  render={<a href="https://github.com/BerriAI/litellm" target="_blank" rel="noopener noreferrer" />}
                >
                  {t("shell.communityButtons.githubLabel")}
                </DropdownMenuItem>
                <DropdownMenuItem
                  render={<a href="https://www.litellm.ai/support" target="_blank" rel="noopener noreferrer" />}
                >
                  {t("shell.communityButtons.slackLabel")}
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </>
  );
}
