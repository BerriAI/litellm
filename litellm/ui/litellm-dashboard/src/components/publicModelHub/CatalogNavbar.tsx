"use client";

import { Tabs } from "@base-ui/react/tabs";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { DOCS_URL } from "@/components/Navbar/DocsLink/DocsLink";
import { useThemeToggle } from "@/components/ThemeToggle/ThemeToggle";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Kbd } from "@/components/ui/kbd";
import { useTheme as useBranding } from "@/contexts/ThemeContext";
import { uiHref } from "@/utils/uiHref";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import { catalogTypography } from "./fonts";
import blueLogo from "./images/litellm-primary-blue.svg";

const SEARCH_SHORTCUTS = "Meta+K Control+K";

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
  const { label: themeLabel, toggleTheme } = useThemeToggle();
  const { logoUrl, logoUrlDark } = useBranding();
  const hideBlog = useDisableBlogPosts();
  const hideCommunity = useDisableShowPrompts();
  const inputRef = useRef<HTMLInputElement>(null);
  const iconButton =
    "grid size-[34px] place-items-center rounded-md border bg-transparent p-0 text-base leading-[1.6] font-normal text-foreground shadow-none hover:border-foreground hover:bg-accent focus-visible:border-border focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-foreground focus-visible:ring-0 active:not-aria-[haspopup]:translate-y-0 aria-expanded:bg-transparent dark:bg-transparent dark:hover:bg-accent [&_svg]:size-[19px]";
  const navTab =
    "flex cursor-pointer items-center border-b-2 border-b-transparent text-[13px] leading-[1.6] font-normal whitespace-nowrap text-muted-foreground hover:border-b-primary hover:text-primary data-active:border-b-primary data-active:text-primary focus-visible:outline-2 focus-visible:outline-offset-4 max-[640px]:min-h-11";
  const menuItem = "cursor-pointer rounded-none px-3 py-2.5 text-[13px] leading-[1.6] font-normal hover:bg-accent";

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
    <header className="relative z-floating border-b border-border bg-background">
      <div
        data-slot="catalog-navbar"
        className="mx-auto flex min-h-[84px] max-w-(--page-width) items-center gap-[34px] px-(--page-gutter) max-[1180px]:gap-6 max-[1000px]:min-h-0 max-[1000px]:flex-wrap max-[1000px]:gap-[18px] max-[1000px]:pt-[18px]"
      >
        <Link
          href={uiHref("")}
          className="flex shrink-0 items-center gap-4 no-underline max-[1000px]:order-0 max-[1000px]:[&_img]:h-7 max-[1000px]:[&_img]:w-[116px]"
        >
          <Image
            src={logoUrl || blueLogo}
            alt={t("shell.navbar.logoAlt")}
            width={132}
            height={32}
            unoptimized
            className="h-8 w-[132px] object-contain dark:hidden"
          />
          <Image
            src={logoUrlDark || logoUrl || blueLogo}
            alt=""
            aria-hidden
            width={132}
            height={32}
            unoptimized
            className={`hidden h-8 w-[132px] object-contain dark:block ${!logoUrlDark && !logoUrl ? "brightness-0 invert" : ""}`}
          />
          <span className="font-mono text-[10px] leading-[1.2] font-normal tracking-[1.4px] whitespace-nowrap text-muted-foreground uppercase max-[1180px]:hidden">
            / {title}
          </span>
        </Link>
        <Tabs.List
          render={<nav />}
          className="flex items-stretch self-stretch gap-[25px] max-[1180px]:gap-5 max-[1000px]:order-3 max-[1000px]:h-11 max-[1000px]:w-full max-[1000px]:gap-7 max-[640px]:h-auto max-[640px]:min-h-11 max-[640px]:flex-wrap max-[640px]:justify-between max-[640px]:gap-x-3 max-[640px]:gap-y-0"
          aria-label={t("publicModelHub.header.title")}
        >
          <Tabs.Tab value="models" className={navTab}>
            {t("publicModelHub.catalog.modelHub")}
          </Tabs.Tab>
          <Tabs.Tab value="skills" className={navTab}>
            {t("publicModelHub.catalog.skillsHub")}
          </Tabs.Tab>
        </Tabs.List>
        <form
          className="ml-auto w-60 min-w-[150px] max-[1000px]:order-2 max-[1000px]:m-0 max-[1000px]:w-full"
          role="search"
          onSubmit={(event) => event.preventDefault()}
        >
          <InputGroup className="h-9 gap-2.5 rounded-md px-[11px] shadow-none focus-within:border-foreground has-[[data-slot=input-group-control]:focus-visible]:ring-0 has-[>[data-align=inline-start]]:[&>input]:pl-0 has-[>[data-align=inline-end]]:[&>input]:pr-0 dark:bg-transparent max-[1000px]:h-[38px]">
            <InputGroupAddon className="p-0">
              <svg
                width="15"
                height="15"
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                aria-hidden="true"
                className="size-[15px] shrink-0"
              >
                <circle cx="8.5" cy="8.5" r="5.5" />
                <path d="m13 13 4 4" />
              </svg>
            </InputGroupAddon>
            <InputGroupInput
              className="h-auto p-0 text-xs leading-[1.6] font-normal placeholder:opacity-100 md:text-xs pointer-coarse:text-base"
              ref={inputRef}
              type="search"
              autoComplete="off"
              aria-label={t("publicModelHub.catalog.searchModels")}
              placeholder={t("publicModelHub.catalog.searchModels")}
              value={search}
              onChange={(event) => onSearchChange(event.target.value)}
              aria-keyshortcuts={SEARCH_SHORTCUTS}
            />
            <InputGroupAddon align="inline-end" className="p-0 has-[>kbd]:mr-0 [&>kbd]:rounded-none">
              <Kbd
                className="h-auto min-w-0 rounded-none bg-transparent p-0 font-mono text-[9px] leading-[normal] font-normal opacity-80 min-[641px]:max-[900px]:hidden"
                aria-hidden="true"
              >
                ⌘ K
              </Kbd>
            </InputGroupAddon>
          </InputGroup>
        </form>
        <div className="flex shrink-0 items-center gap-4 max-[1180px]:gap-2.5 max-[1000px]:order-1 max-[1000px]:ml-auto min-[641px]:max-[900px]:gap-1.5">
          <a
            href={DOCS_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs leading-[1.6] whitespace-nowrap text-muted-foreground no-underline hover:text-foreground max-[1180px]:hidden"
            aria-label={t("shell.docsLink.label")}
          >
            {`${t("shell.docsLink.label")} ↗`}
          </a>
          <Button
            type="button"
            variant="outline"
            size="icon"
            className={iconButton}
            aria-label={themeLabel}
            title={themeLabel}
            onClick={toggleTheme}
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="size-[19px] dark:hidden"
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
              className="hidden size-[19px] dark:block"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
            </svg>
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className={`${iconButton} pb-2.5 text-[25px] leading-none tracking-[1px]`}
                  aria-label={t("publicModelHub.catalog.moreNavigation")}
                />
              }
            >
              <span aria-hidden="true">···</span>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              sideOffset={11}
              className={`${catalogTypography} min-w-[210px] rounded-lg border border-border bg-card p-2 shadow-[0_12px_30px_rgb(0_0_0/16%)] ring-0`}
            >
              <DropdownMenuItem
                className={menuItem}
                render={<a href={DOCS_URL} target="_blank" rel="noopener noreferrer" />}
              >
                {t("shell.docsLink.label")}
              </DropdownMenuItem>
              {!hideBlog && (
                <DropdownMenuItem
                  className={menuItem}
                  render={<a href="https://docs.litellm.ai/blog" target="_blank" rel="noopener noreferrer" />}
                >
                  {t("shell.blogDropdown.trigger")}
                </DropdownMenuItem>
              )}
              {!hideCommunity && (
                <>
                  <DropdownMenuItem
                    className={menuItem}
                    render={<a href="https://github.com/BerriAI/litellm" target="_blank" rel="noopener noreferrer" />}
                  >
                    {t("shell.communityButtons.githubLabel")}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className={menuItem}
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
