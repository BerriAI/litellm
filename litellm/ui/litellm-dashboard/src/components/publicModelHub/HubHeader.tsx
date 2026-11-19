"use client";

import { ChevronDown, ExternalLink } from "lucide-react";
import { useId } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Field, FieldDescription } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";

import { gatewayUrl, usefulLinkList } from "./hubFormat";
import { CopyIconButton } from "./HubParts";
import type { HubTab } from "./useHubUrlState";
import styles from "./catalog.module.css";
import type { HubInfo } from "./useHubData";

export type HubStatus = "loading" | "operational" | "degraded" | "unavailable";

function LinksMenu({ info }: { info: HubInfo | undefined }) {
  const { t } = useTranslation();
  const links = usefulLinkList(info?.useful_links);
  if (links.length === 0) return null;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<Button variant="outline" size="sm" />}>
        {t("publicModelHub.linksMenu.trigger")}
        <ChevronDown className="text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-auto min-w-(--anchor-width)">
        {links.map((link) => (
          <DropdownMenuItem key={link.title} render={<a href={link.url} target="_blank" rel="noopener noreferrer" />}>
            <ExternalLink />
            {link.title}
          </DropdownMenuItem>
        ))}
        {info?.litellm_version && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuLabel className="text-xs font-normal">
                {t("publicModelHub.linksMenu.version", { version: info.litellm_version })}
              </DropdownMenuLabel>
            </DropdownMenuGroup>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function BaseUrlField() {
  const { t } = useTranslation();
  const url = gatewayUrl();
  return (
    <Field className="w-full gap-1.5 sm:w-[360px]">
      <InputGroup className="h-8">
        <InputGroupAddon>
          <InputGroupText className="text-xs font-medium">{t("publicModelHub.baseUrlField.label")}</InputGroupText>
        </InputGroupAddon>
        <InputGroupInput
          readOnly
          value={url}
          aria-label={t("publicModelHub.baseUrlField.label")}
          aria-describedby="hub-base-url-hint"
          className="font-mono text-xs md:text-xs"
        />
        <InputGroupAddon align="inline-end">
          <CopyIconButton value={url} label={t("publicModelHub.baseUrlField.copy")} className="size-6" />
        </InputGroupAddon>
      </InputGroup>
      <FieldDescription id="hub-base-url-hint" className="hidden text-xs text-balance sm:block">
        {t("publicModelHub.baseUrlField.hint")}
      </FieldDescription>
    </Field>
  );
}

export function HubHeader({ info, status, tab }: { info: HubInfo | undefined; status: HubStatus; tab: HubTab }) {
  const { t } = useTranslation();
  const gridId = useId();
  return (
    <header className={styles.header}>
      <div>
        <h1 className={styles.title}>
          {t(tab === "skills" ? "publicModelHub.catalog.skillsHub" : "publicModelHub.catalog.modelHub")}
        </h1>
        <p className={`${styles.description} m-0 max-w-[520px] text-[17px] tracking-[-0.3px]`}>
          {info?.custom_docs_description ||
            t(
              tab === "skills"
                ? "publicModelHub.catalog.skillsDescription"
                : "publicModelHub.catalog.modelsDescription",
            )}
        </p>
        {status !== "unavailable" && (
          <details className={styles.quickStart}>
            <summary>{t("publicModelHub.catalog.quickStart")}</summary>
            <div className="flex flex-wrap items-start gap-4 py-3">
              <BaseUrlField />
              <LinksMenu info={info} />
            </div>
          </details>
        )}
      </div>
      <svg className={styles.blueprint} viewBox="0 0 360 220" fill="none" aria-hidden="true">
        <defs>
          <pattern id={gridId} width="32" height="32" patternUnits="userSpaceOnUse">
            <path d="M32 0H0V32" stroke="currentColor" strokeWidth=".7" opacity=".09" />
          </pattern>
        </defs>
        <rect x="20" y="14" width="320" height="192" fill={`url(#${gridId})`} />
        <g stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <rect x="66" y="35" width="228" height="150" rx="8" className="fill-background" />
          <path opacity=".3" d="M66 70h228M66 108h228M66 146h228M208 70v115" />
          <path d="M88 52h72M88 89h92M88 127h66M88 165h80M230 89h40M230 127h40M230 165h40" />
          <circle cx="268" cy="52" r="3" fill="currentColor" stroke="none" />
        </g>
      </svg>
    </header>
  );
}
