"use client";

import { Copy, Search, X } from "lucide-react";
import * as React from "react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Field, FieldLabel } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { ItemTitle } from "@/components/ui/item";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { providerLogo, providerName, SECTION_TITLE } from "./hubFormat";

export function CopyIconButton({ value, label, className }: { value: string; label: string; className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={label}
            className={cn("text-muted-foreground", className)}
            onClick={() => void copyToClipboard(value, "Copied")}
          />
        }
      >
        <Copy />
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

export function CodeBlock({ code, label = "Copy code" }: { code: string; label?: string }) {
  return (
    <div className="relative min-w-0 rounded-lg border border-border bg-muted">
      <CopyIconButton value={code} label={label} className="absolute top-1.5 right-1.5 bg-background" />
      <pre className="max-h-96 overflow-auto p-3 pr-12 font-mono text-xs leading-5 whitespace-pre text-foreground">
        {code}
      </pre>
    </div>
  );
}

export function ProviderAvatar({ provider, className }: { provider: string; className?: string }) {
  const logo = providerLogo(provider);
  return (
    <Avatar className={cn("size-4 bg-white ring-1 ring-border", className)}>
      {logo && <AvatarImage src={logo} alt="" className="object-contain p-0.5" />}
      <AvatarFallback className="bg-muted text-xs">{providerName(provider).charAt(0)}</AvatarFallback>
    </Avatar>
  );
}

export function ProviderName({ provider }: { provider: string }) {
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <ProviderAvatar provider={provider} />
      <span className="truncate">{providerName(provider)}</span>
    </span>
  );
}

export function Badges({ items, mono }: { items: string[]; mono?: boolean }) {
  if (items.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((item) => (
        <Badge key={item} variant="secondary" className={cn("font-medium", mono && "font-mono font-normal")}>
          {item}
        </Badge>
      ))}
    </div>
  );
}

export function PaneSection({
  title,
  aside,
  children,
}: {
  title: string;
  aside?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className={SECTION_TITLE}>{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

export type Fact = { label: string; value: React.ReactNode; wide?: boolean };

const BLANK: React.ReactNode[] = [null, undefined, false, "", "-"];

const present = (value: React.ReactNode) => !BLANK.includes(value);

export function Facts({ groups }: { groups: Fact[][] }) {
  const shown = groups.map((group) => group.filter((fact) => present(fact.value))).filter((group) => group.length);
  if (shown.length === 0) return null;
  return (
    <div className="flex flex-col gap-4">
      {shown.map((group) => (
        <dl key={group[0].label} className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
          {group.map((fact) => (
            <div key={fact.label} className={cn("flex min-w-0 flex-col gap-1", fact.wide && "col-span-2")}>
              <dt className="text-xs font-medium whitespace-nowrap text-muted-foreground">{fact.label}</dt>
              <dd className="min-w-0 truncate text-sm tabular-nums">{fact.value}</dd>
            </div>
          ))}
        </dl>
      ))}
    </div>
  );
}

export function EndpointField({ id, value }: { id: string; value: string }) {
  return (
    <Field>
      <FieldLabel htmlFor={id}>Endpoint</FieldLabel>
      <InputGroup>
        <InputGroupInput id={id} readOnly value={value} className="font-mono text-xs md:text-xs" />
        <InputGroupAddon align="inline-end">
          <CopyIconButton value={value} label="Copy endpoint" className="size-6" />
        </InputGroupAddon>
      </InputGroup>
    </Field>
  );
}

export function SearchInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}) {
  return (
    <InputGroup className="h-8 min-w-0 flex-1">
      <InputGroupAddon>
        <Search />
      </InputGroupAddon>
      <InputGroupInput
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        aria-label={placeholder}
      />
      {value && (
        <InputGroupAddon align="inline-end">
          <InputGroupButton size="icon-xs" aria-label="Clear search" onClick={() => onChange("")}>
            <X />
          </InputGroupButton>
        </InputGroupAddon>
      )}
    </InputGroup>
  );
}

export function RowTitle({ children }: { children: React.ReactNode }) {
  return (
    <ItemTitle className="w-full min-w-0">
      <span className="line-clamp-2 wrap-anywhere">{children}</span>
    </ItemTitle>
  );
}

export function Joined({ parts }: { parts: (string | null | undefined)[] }) {
  return parts.filter(Boolean).map((part, index) => (
    <React.Fragment key={part}>
      {index > 0 && " · "}
      <span className="whitespace-nowrap">{part}</span>
    </React.Fragment>
  ));
}
