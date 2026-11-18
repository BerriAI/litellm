"use client";

import { ChevronDown, ExternalLink } from "lucide-react";
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
  return (
    <header className={styles.header}>
      <div>
        <h1 className={styles.title}>
          {t(tab === "skills" ? "publicModelHub.catalog.skillsHub" : "publicModelHub.catalog.modelHub")}
        </h1>
        <p className={styles.description}>
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
            <div className={styles.connection}>
              <BaseUrlField />
              <LinksMenu info={info} />
            </div>
          </details>
        )}
      </div>
      <svg className={styles.blueprint} viewBox="0 0 390 208" fill="none" aria-hidden="true">
        <path
          opacity=".09"
          stroke="currentColor"
          strokeWidth=".7"
          d="M40 8v192M72 8v192M104 8v192M136 8v192M168 8v192M200 8v192M232 8v192M264 8v192M296 8v192M328 8v192M16 26h352M16 58h352M16 90h352M16 122h352M16 154h352M16 186h352"
        />
        <g stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <rect x="62" y="28" width="242" height="158" rx="8" className={styles.blueprintSurface} />
          <path opacity=".3" d="M62 65h242M62 105h242M62 145h242M212 65v121" />
          <path d="M85 46h76M85 85h98M235 85h42M85 125h70M235 125h42M85 165h85M235 165h42" />
          <circle cx="276" cy="46" r="3" fill="currentColor" stroke="none" />
        </g>
      </svg>
    </header>
  );
}
