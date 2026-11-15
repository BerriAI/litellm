import { useState } from "react";
import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { Check, Copy } from "lucide-react";
import { useTranslation } from "react-i18next";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { createApiClient } from "@/lib/http/client";
import { getAuthHeaderName } from "@/lib/http/runtime";
import { buildSkillInstallCommand, findPublicSkill, SKILL_AGENTS, type SkillAgent } from "./skillInstall";

interface SkillInstallDialogProps {
  name: string;
  archivePath: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function SkillInstallDialog({ name, archivePath, open, onOpenChange }: SkillInstallDialogProps) {
  const { t } = useTranslation();
  const [agent, setAgent] = useState<SkillAgent>("codex");
  const [scope, setScope] = useState("project");
  const [mode, setMode] = useState("authenticated");
  const [copyState, setCopyState] = useState<{
    command: string;
    status: "pending" | "copied" | "failed";
  } | null>(null);
  const gatewayUrl = getProxyBaseUrl().replace(/\/+$/, "");
  const archiveUrl = `${gatewayUrl}${archivePath}`;
  const indexOptions: UseQueryOptions<unknown> = {
    queryKey: ["public-skill-install", gatewayUrl, archivePath],
    queryFn: ({ signal }) =>
      createApiClient({ getBaseUrl: () => gatewayUrl }).get<unknown>("/.well-known/agent-skills/index.json", {
        signal,
        credentials: "omit",
        redirect: "error",
      }),
    enabled: open,
    retry: false,
    staleTime: Infinity,
    gcTime: 0,
    refetchOnMount: "always",
  };
  const index = useQuery(indexOptions);
  const publicSkill = index.isSuccess && !index.isFetching ? findPublicSkill(index.data, archiveUrl) : undefined;
  const publicInstall = mode === "public" && publicSkill !== undefined;
  const commandOptions = {
    gatewayUrl,
    archiveUrl,
    name,
    agent,
    global: scope === "global",
    publicSkill: publicInstall ? publicSkill : undefined,
    authHeaderName: getAuthHeaderName(),
  };
  const command = buildSkillInstallCommand(commandOptions);
  const copied = copyState?.command === command && copyState.status === "copied";
  const copyPending = copyState?.command === command && copyState.status === "pending";
  const copyFailed = copyState?.command === command && copyState.status === "failed";
  const copyCommand = async () => {
    if (!command) return;
    setCopyState({ command, status: "pending" });
    try {
      await navigator.clipboard.writeText(command);
      setCopyState({ command, status: "copied" });
    } catch {
      setCopyState({ command, status: "failed" });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90dvh] min-w-0 overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t("skills.install.title", { name })}</DialogTitle>
          <DialogDescription>{t("skills.install.description")}</DialogDescription>
        </DialogHeader>
        <div className="grid min-w-0 gap-5 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="skill-install-agent">{t("skills.install.agent")}</Label>
            <Select<SkillAgent> items={SKILL_AGENTS} value={agent} onValueChange={(value) => value && setAgent(value)}>
              <SelectTrigger id="skill-install-agent" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SKILL_AGENTS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium">{t("skills.install.scope")}</p>
            <RadioGroup aria-label={t("skills.install.scope")} value={scope} onValueChange={setScope}>
              <Label>
                <RadioGroupItem value="project" />
                {t("skills.install.project")}
              </Label>
              <Label>
                <RadioGroupItem value="global" />
                {t("skills.install.global")}
              </Label>
            </RadioGroup>
          </div>
        </div>
        {publicSkill && (
          <RadioGroup
            aria-label={t("skills.install.method")}
            value={publicInstall ? "public" : "authenticated"}
            onValueChange={setMode}
            className="flex flex-wrap gap-5"
          >
            <Label>
              <RadioGroupItem value="authenticated" />
              {t("skills.install.authenticated")}
            </Label>
            <Label>
              <RadioGroupItem value="public" />
              {t("skills.install.public")}
            </Label>
          </RadioGroup>
        )}
        <div className="min-w-0 space-y-3">
          <p className="text-sm text-muted-foreground">
            {t(publicInstall ? "skills.install.publicHint" : "skills.install.authenticatedHint")}
          </p>
          {command ? (
            <pre
              aria-label={t("skills.install.command")}
              className="max-h-72 overflow-auto rounded-lg border border-border bg-muted p-4 text-xs leading-relaxed"
              tabIndex={0}
            >
              <code>{command}</code>
            </pre>
          ) : (
            <p role="alert" className="text-sm text-destructive">
              {t("skills.install.unavailable")}
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            {t(publicInstall ? "skills.install.publicRequirements" : "skills.install.authenticatedRequirements")}
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">{t("skills.install.shareHint")}</p>
          <Button onClick={copyCommand} disabled={!command || copyPending}>
            {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
            {t(copied ? "skills.install.copied" : "skills.install.copy")}
          </Button>
        </div>
        {copyFailed && (
          <p role="alert" className="text-sm text-destructive">
            {t("skills.install.copyFailed")}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
