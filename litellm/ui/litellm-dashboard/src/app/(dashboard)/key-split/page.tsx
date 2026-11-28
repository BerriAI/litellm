"use client";

import { useTranslation } from "react-i18next";
import React, { useEffect, useState } from "react";
import { Split } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface InferredAgent {
  fingerprint: string;
  suggested_name: string;
  system_prompt: string;
  tools: string[];
  models: string[];
  requests: number;
  failures: number;
  spend: number;
}
interface SharedKey {
  key_hash: string;
  key_alias: string | null;
  requests: number;
  spend: number;
  agents: InferredAgent[];
}
interface KeySplit {
  keys_scanned: number;
  shared_keys: SharedKey[];
}

export default function KeySplitPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [data, setData] = useState<KeySplit | null>(null);
  const [minted, setMinted] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!accessToken) return;
    apiClient.get<KeySplit>("/what_if/key_split", { accessToken }).then(setData);
  }, [accessToken]);

  const mint = async (shared: SharedKey, agent: InferredAgent) => {
    const created = await apiClient.post<{ key: string }>("/key/generate", {
      accessToken: accessToken ?? undefined,
      body: {
        key_alias: `${agent.suggested_name}-${agent.fingerprint}`,
        models: agent.models,
        metadata: { split_from: shared.key_alias ?? shared.key_hash, agent_fingerprint: agent.fingerprint },
      },
    });
    setMinted((keys) => ({ ...keys, [agent.fingerprint]: created.key }));
  };

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <Split />
          {t("whatIf.keySplit.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.keySplit.description")}</PageHeaderDescription>
      </PageHeader>
      {data?.shared_keys.map((shared) => (
        <Card key={shared.key_hash}>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <code>{shared.key_alias ?? shared.key_hash.slice(0, 12)}</code>
              <Badge variant="destructive">{t("whatIf.keySplit.agentsBehind", { count: shared.agents.length })}</Badge>
              <span className="text-sm font-normal text-muted-foreground">
                {t("whatIf.keySplit.totals", { requests: shared.requests, spend: shared.spend.toFixed(4) })}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("whatIf.keySplit.agent")}</TableHead>
                  <TableHead>{t("whatIf.keySplit.tools")}</TableHead>
                  <TableHead>{t("whatIf.keySplit.model")}</TableHead>
                  <TableHead>{t("whatIf.keySplit.requests")}</TableHead>
                  <TableHead>{t("whatIf.keySplit.failed")}</TableHead>
                  <TableHead>{t("whatIf.keySplit.spend")}</TableHead>
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {shared.agents.map((agent) => (
                  <TableRow key={agent.fingerprint}>
                    <TableCell>
                      <div className="font-medium">{agent.suggested_name}</div>
                      <div className="text-xs text-muted-foreground">{agent.system_prompt.slice(0, 60)}</div>
                    </TableCell>
                    <TableCell>
                      <code className="text-xs">{agent.tools.join(", ") || "-"}</code>
                    </TableCell>
                    <TableCell>{agent.models.join(", ")}</TableCell>
                    <TableCell className="font-semibold">{agent.requests}</TableCell>
                    <TableCell>
                      {agent.failures > 0 ? <Badge variant="destructive">{agent.failures}</Badge> : agent.failures}
                    </TableCell>
                    <TableCell>${agent.spend.toFixed(4)}</TableCell>
                    <TableCell>
                      {minted[agent.fingerprint] ? (
                        <code className="text-xs">{minted[agent.fingerprint].slice(0, 14)}...</code>
                      ) : (
                        <Button size="sm" variant="outline" onClick={() => mint(shared, agent)}>
                          {t("whatIf.keySplit.mint")}
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ))}
      {data && data.shared_keys.length === 0 && (
        <Card>
          <CardContent>{t("whatIf.keySplit.none", { count: data.keys_scanned })}</CardContent>
        </Card>
      )}
    </Page>
  );
}
