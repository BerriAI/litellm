import type { ParseKeys } from "i18next";
import React from "react";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import {
  ArrowLeft,
  ArrowLeftRight,
  Ban,
  Building2,
  CircleCheck,
  Ellipsis,
  Folder,
  KeyRound,
  Plus,
  RefreshCw,
  Trash2,
  User,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import CopyButton from "@/components/shared/CopyButton";
import { StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { orgDetailHref, teamDetailHref, userDetailHref } from "@/utils/entityLinks";
import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import { middleTruncate } from "./key_detail/formatters";
import { ResourceLink } from "./key_detail/ResourceLink";

export type KeyStatus = "active" | "blocked" | "overBudget" | "expired";

export interface KeyInfoData {
  keyName: string;
  keyId: string;
  secretKey: string;
  userId: string;
  userEmail: string;
  userAlias?: string | null;
  teamId: string;
  teamAlias?: string | null;
  orgId: string;
  orgAlias?: string | null;
  projectId?: string | null;
  projectAlias?: string | null;
}

interface KeyInfoHeaderProps {
  data: KeyInfoData;
  status?: KeyStatus;
  onBack?: () => void;
  onCreateNew?: () => void;
  onRegenerate?: () => void;
  onDelete?: () => void;
  onResetSpend?: () => void;
  onToggleBlocked?: () => void;
  isBlocked?: boolean;
  canModifyKey?: boolean;
  backButtonText?: string;
  regenerateDisabled?: boolean;
  regenerateTooltip?: string;
  tabs?: React.ReactNode;
}

const STATUS_TONE: Readonly<Record<KeyStatus, StatusTone>> = {
  active: "success",
  blocked: "neutral",
  overBudget: "error",
  expired: "error",
};

function MetaItem({ icon: Icon, label, children }: { icon?: typeof User; label?: string; children: React.ReactNode }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1.5" title={label}>
      {Icon ? <Icon className="size-4 shrink-0" aria-hidden /> : null}
      {children}
    </span>
  );
}

function IdChip({ value, label, copyLabel }: { value: string; label: string; copyLabel?: string }) {
  return (
    <span className="inline-flex items-center gap-0.5">
      <span
        className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-xs text-foreground"
        title={value}
        aria-label={label}
      >
        {middleTruncate(value)}
      </span>
      {copyLabel ? <CopyButton value={value} label={copyLabel} iconClassName="size-3.5" /> : null}
    </span>
  );
}

function UserField({ userAlias, userEmail, userId }: { userAlias?: string | null; userEmail: string; userId: string }) {
  const { t } = useTranslation();
  if (!userAlias && !userEmail && !userId) {
    return <span>{t("keyDetail.keyInfoHeader.noOwner")}</span>;
  }

  const isDefaultAdmin = userId === DEFAULT_PROXY_ADMIN_USER_ID;
  const displayValue = userAlias || userEmail || userId;

  const popoverContent = (
    <div className="flex min-w-[200px] max-w-[300px] flex-col gap-2 text-xs">
      {[
        { label: t("keyDetail.keyInfoHeader.user.alias"), value: userAlias ?? null },
        { label: t("keyDetail.keyInfoHeader.user.email"), value: userEmail || null },
        { label: t("keyDetail.keyInfoHeader.user.id"), value: userId || null },
      ].map(({ label, value }) => (
        <div key={label} className="flex min-w-0 flex-col">
          <span className="text-muted-foreground">{label}</span>
          {value ? (
            <div className="flex min-w-0 items-center gap-1">
              <span className="min-w-0 flex-1 truncate font-mono text-xs" title={value}>
                {value}
              </span>
              <CopyButton
                value={value}
                label={t("keyDetail.keyInfoHeader.user.copyField", { label })}
                iconClassName="size-3.5"
              />
            </div>
          ) : (
            <span className="font-mono">-</span>
          )}
        </div>
      ))}
    </div>
  );

  const trigger =
    isDefaultAdmin && !userAlias && !userEmail ? (
      <span className="cursor-default">
        <DefaultProxyAdminTag userId={userId} />
      </span>
    ) : (
      <span className="block max-w-[240px] cursor-default truncate">
        <ResourceLink href={userId ? userDetailHref(userId) : undefined}>{displayValue}</ResourceLink>
      </span>
    );

  return (
    <HoverCard>
      <HoverCardTrigger render={trigger} />
      <HoverCardContent side="bottom" align="start" className="w-auto">
        {popoverContent}
      </HoverCardContent>
    </HoverCard>
  );
}

export function KeyInfoHeader({
  data,
  status = "active",
  onBack,
  onCreateNew,
  onRegenerate,
  onDelete,
  onResetSpend,
  onToggleBlocked,
  isBlocked = false,
  canModifyKey = true,
  backButtonText = i18n.t("keyDetail.keyInfoHeader.backToKeys"),
  regenerateDisabled = false,
  regenerateTooltip,
  tabs,
}: KeyInfoHeaderProps) {
  const { t } = useTranslation();
  const regenerateButton = (
    <span>
      <Button variant="outline" size="sm" onClick={onRegenerate} disabled={regenerateDisabled}>
        <RefreshCw className="size-3.5" />
        {t("keyDetail.keyInfoHeader.regenerateKey")}
      </Button>
    </span>
  );

  return (
    <header>
      {onCreateNew && (
        <div className="mb-4">
          <Button onClick={onCreateNew}>
            <Plus className="size-3.5" />
            {t("keyDetail.keyInfoHeader.createNewKey")}
          </Button>
        </div>
      )}

      <button
        type="button"
        onClick={onBack}
        className="mb-2 inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3.5" />
        {backButtonText}
      </button>

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1.5">
          <div className="flex min-w-0 items-center gap-2.5">
            <h1 className="m-0 flex min-w-0 items-center gap-2 text-xl leading-7 font-semibold tracking-[-0.01em]">
              <KeyRound className="size-5 shrink-0" strokeWidth={1.75} aria-hidden />
              <span className="min-w-0 truncate">{data.keyName}</span>
            </h1>
            <CopyButton value={data.keyName} label={t("keyDetail.keyInfoHeader.copyKeyAlias")} iconClassName="size-4" />
            <StatusBadge
              tone={STATUS_TONE[status]}
              label={t(`keyDetail.keyInfoHeader.status.${status}` as ParseKeys)}
            />
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] leading-5 text-muted-foreground">
            <MetaItem icon={User} label={t("keyDetail.keyInfoHeader.user.label")}>
              <UserField userAlias={data.userAlias} userEmail={data.userEmail} userId={data.userId} />
            </MetaItem>
            {data.teamId ? (
              <MetaItem icon={Users} label={t("keyDetail.keyInfoHeader.fields.team")}>
                <ResourceLink href={teamDetailHref(data.teamId)}>{data.teamAlias || data.teamId}</ResourceLink>
              </MetaItem>
            ) : null}
            {data.orgId ? (
              <MetaItem icon={Building2} label={t("keyDetail.keyInfoHeader.fields.organization")}>
                <ResourceLink href={orgDetailHref(data.orgId)}>{data.orgAlias || data.orgId}</ResourceLink>
              </MetaItem>
            ) : null}
            {data.projectId ? (
              <MetaItem icon={Folder} label={t("keyDetail.keyInfoHeader.fields.project")}>
                <span className="font-medium text-foreground">{data.projectAlias || data.projectId}</span>
              </MetaItem>
            ) : null}
            <MetaItem>
              {data.secretKey ? <IdChip value={data.secretKey} label={t("keyDetail.keyInfoHeader.secretKey")} /> : null}
              <IdChip
                value={data.keyId}
                label={t("keyDetail.keyInfoHeader.keyIdLabel")}
                copyLabel={t("keyDetail.keyInfoHeader.copyKeyId")}
              />
            </MetaItem>
          </div>
        </div>
        {canModifyKey && (
          <div className="flex items-center gap-2">
            {regenerateTooltip ? (
              <TooltipProvider delay={300}>
                <Tooltip>
                  <TooltipTrigger render={regenerateButton} />
                  <TooltipContent>{regenerateTooltip}</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            ) : (
              regenerateButton
            )}
            <DropdownMenu>
              <DropdownMenuTrigger
                render={
                  <Button variant="outline" size="icon-sm" aria-label={t("keyDetail.keyInfoHeader.moreActions")} />
                }
              >
                <Ellipsis className="size-4" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-auto">
                {onToggleBlocked && isBlocked ? (
                  <DropdownMenuItem onClick={onToggleBlocked}>
                    <CircleCheck className="size-3.5" />
                    {t("keyDetail.keyInfoHeader.actions.unblock")}
                  </DropdownMenuItem>
                ) : null}
                {onToggleBlocked && !isBlocked ? (
                  <DropdownMenuItem variant="destructive" onClick={onToggleBlocked}>
                    <Ban className="size-3.5" />
                    {t("keyDetail.keyInfoHeader.actions.block")}
                  </DropdownMenuItem>
                ) : null}
                {onResetSpend && (
                  <DropdownMenuItem variant="destructive" onClick={onResetSpend}>
                    <ArrowLeftRight className="size-3.5" />
                    {t("keyDetail.keyInfoHeader.actions.resetSpend")}
                  </DropdownMenuItem>
                )}
                {onToggleBlocked || onResetSpend ? <DropdownMenuSeparator /> : null}
                <DropdownMenuItem variant="destructive" onClick={onDelete}>
                  <Trash2 className="size-3.5" />
                  {t("keyDetail.keyInfoHeader.actions.delete")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </div>

      {tabs ? <div className="mt-4">{tabs}</div> : null}
    </header>
  );
}
