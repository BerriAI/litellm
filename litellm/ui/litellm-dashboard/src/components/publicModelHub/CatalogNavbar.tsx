"use client";

import { Tabs } from "@base-ui/react/tabs";
import Image from "next/image";
import Link from "next/link";
import { useTheme } from "next-themes";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { DOCS_URL } from "@/components/Navbar/DocsLink/DocsLink";
import { useTheme as useBranding } from "@/contexts/ThemeContext";
import { uiHref } from "@/utils/uiHref";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import styles from "./catalog.module.css";
import blueLogo from "./images/litellm-primary-blue.svg";
import whiteLogo from "./images/litellm-primary-white.svg";

export function CatalogNavbar({
  title,
  search,
  onSearchChange,
}: {
  title: string;
  search: string;
  onSearchChange: (value: string) => void;
}) {
  const { t } = useTranslation();
  const { resolvedTheme, setTheme } = useTheme();
  const { logoUrl, logoUrlDark } = useBranding();
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
    <header className={styles.navbar}>
      <div className={styles.navbarInner}>
        <Link href={uiHref("")} className={styles.brand}>
          <Image
            src={logoUrl || blueLogo}
            alt={t("shell.navbar.logoAlt")}
            width={132}
            height={32}
            unoptimized
            className="dark:hidden"
          />
          <Image
            src={logoUrlDark || logoUrl || whiteLogo}
            alt=""
            aria-hidden
            width={132}
            height={32}
            unoptimized
            className="hidden dark:block"
          />
          <span>/ {title}</span>
        </Link>
        <Tabs.List render={<nav />} className={styles.navigation} aria-label={t("publicModelHub.header.title")}>
          <Tabs.Tab value="models">{t("publicModelHub.catalog.modelHub")}</Tabs.Tab>
          <Tabs.Tab value="skills">{t("publicModelHub.catalog.skillsHub")}</Tabs.Tab>
        </Tabs.List>
        <form className={styles.navSearch} role="search" onSubmit={(event) => event.preventDefault()}>
          <svg
            width="15"
            height="15"
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            aria-hidden="true"
          >
            <circle cx="8.5" cy="8.5" r="5.5" />
            <path d="m13 13 4 4" />
          </svg>
          <input
            ref={inputRef}
            type="search"
            autoComplete="off"
            aria-label={t("publicModelHub.catalog.searchModels")}
            placeholder={t("publicModelHub.catalog.searchModels")}
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            aria-keyshortcuts="Meta+K Control+K"
          />
          <kbd aria-hidden="true">⌘ K</kbd>
        </form>
        <div className={styles.navActions}>
          <a
            href={DOCS_URL}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.navDocs}
            aria-label={t("shell.docsLink.label")}
          >
            {t("shell.docsLink.label")} ↗
          </a>
          <button
            type="button"
            className={styles.navIconButton}
            aria-label={themeLabel}
            title={themeLabel}
            onClick={() => setTheme(isDark ? "light" : "dark")}
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="dark:hidden"
              aria-hidden="true"
            >
              <path d="M20.8 13.1A9 9 0 0 1 10.9 3.2 9 9 0 1 0 20.8 13.1Z" />
            </svg>
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="hidden dark:block"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
            </svg>
          </button>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <button
                  type="button"
                  className={`${styles.navIconButton} ${styles.moreToggle}`}
                  aria-label={t("publicModelHub.catalog.moreNavigation")}
                />
              }
            >
              <span aria-hidden="true">···</span>
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
      </div>
    </header>
  );
}
