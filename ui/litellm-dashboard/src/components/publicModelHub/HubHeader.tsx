"use client";

import { ChevronDown, ExternalLink, LayoutGrid } from "lucide-react";
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
import { StatusBadge } from "./HubStatus";
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
    <Field className="w-full gap-1.5 sm:w-[420px]">
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
      <FieldDescription id="hub-base-url-hint" className="text-xs text-balance">
        {t("publicModelHub.baseUrlField.hint")}
      </FieldDescription>
    </Field>
  );
}

export function HubHeader({ info, status }: { info: HubInfo | undefined; status: HubStatus }) {
  const { t } = useTranslation();
  return (
    <header className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <h1 className="flex items-center gap-2 text-xl leading-7 font-semibold">
            <LayoutGrid className="size-5" />
            {t("publicModelHub.header.title")}
          </h1>
          {status === "operational" && <StatusBadge tone="success" label={t("publicModelHub.header.operational")} />}
        </div>
        <p className="text-sm text-muted-foreground">
          {info?.custom_docs_description || t("publicModelHub.header.description")}
        </p>
      </div>
      {status !== "unavailable" && (
        <div className="flex flex-wrap items-start gap-2">
          <BaseUrlField />
          <LinksMenu info={info} />
        </div>
      )}
    </header>
  );
}
