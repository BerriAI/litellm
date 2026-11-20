"use client";

import { ChevronDown, ExternalLink } from "lucide-react";
import { useId } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
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
    <header
      data-slot="catalog-hero"
      className="grid min-h-[280px] grid-cols-[minmax(0,1fr)_390px] items-start gap-[30px] pt-10 pb-8 catalog-narrow:min-[800px]:grid-cols-[minmax(0,1fr)_300px] max-[800px]:grid-cols-[minmax(0,1fr)_230px] max-[800px]:gap-2.5 max-[640px]:relative max-[640px]:block max-[640px]:min-h-52 max-[640px]:overflow-hidden max-[640px]:py-8"
    >
      <div className="min-w-0 max-[640px]:relative max-[640px]:z-raised">
        <h1 className="mt-0 mb-5 font-[family-name:var(--font-catalog-display),sans-serif] text-[96px] leading-[0.9] font-semibold tracking-[-2px] uppercase catalog-narrow:min-[800px]:text-[81px] max-[800px]:text-[68px] max-[640px]:text-[clamp(52px,10.6vw,68px)] max-[640px]:tracking-[-1px]">
          {t(tab === "skills" ? "publicModelHub.catalog.skillsHub" : "publicModelHub.catalog.modelHub")}
        </h1>
        <p
          data-slot="catalog-description"
          className="m-0 max-w-[520px] text-[17px] tracking-[-0.3px] max-[800px]:text-sm max-[640px]:max-w-80"
        >
          {info?.custom_docs_description ||
            t(
              tab === "skills"
                ? "publicModelHub.catalog.skillsDescription"
                : "publicModelHub.catalog.modelsDescription",
            )}
        </p>
        {status !== "unavailable" && (
          <Collapsible className="mt-[22px] text-[13px]">
            <CollapsibleTrigger
              nativeButton={false}
              render={<span />}
              data-slot="catalog-quick-start"
              className="list-item w-fit list-inside cursor-pointer py-2 font-medium text-primary [list-style-type:disclosure-closed] data-open:[list-style-type:disclosure-open] focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary"
            >
              {t("publicModelHub.catalog.quickStart")}
            </CollapsibleTrigger>
            <CollapsibleContent className="flex flex-wrap items-start gap-4 py-3">
              <BaseUrlField />
              <LinksMenu info={info} />
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
      <svg
        data-slot="catalog-blueprint"
        className="h-52 w-full text-primary max-[640px]:absolute max-[640px]:top-[23px] max-[640px]:right-[-70px] max-[640px]:w-[235px] max-[640px]:opacity-[0.14]"
        viewBox="0 0 360 220"
        fill="none"
        aria-hidden="true"
      >
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
