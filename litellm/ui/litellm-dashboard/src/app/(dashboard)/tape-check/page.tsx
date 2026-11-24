"use client";

import { useTranslation } from "react-i18next";
import React, { useState } from "react";
import { Disc3 } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface Exchange {
  name: string;
  status: number;
  answer: { tools: Record<string, string> };
}
interface Tape {
  recorded_from: string;
  version: string;
  exchanges: Exchange[];
}
interface TapeCheck {
  recorded_version: string;
  candidate_version: string;
  changed: number;
  exchanges: { name: string; verdict: "same" | "changed"; changes: string[] }[];
}

const EMPTY_FORM = { current: "", candidate: "", key: "", header: "", credential: "", calls: "" };

export default function TapeCheckPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [form, setForm] = useState(EMPTY_FORM);
  const [tape, setTape] = useState<Tape | null>(null);
  const [check, setCheck] = useState<TapeCheck | null>(null);

  const field = (name: keyof typeof form, secret = false) => (
    <Input
      aria-label={t(`whatIf.tapeCheck.form.${name}`)}
      placeholder={t(`whatIf.tapeCheck.form.${name}`)}
      type={secret ? "password" : "text"}
      value={form[name]}
      onChange={(event) => setForm((values) => ({ ...values, [name]: event.target.value }))}
    />
  );
  const shared = {
    api_key: form.key,
    upstream_headers: form.header ? { [form.header]: form.credential } : {},
  };
  const record = async () => {
    setCheck(null);
    setTape(
      await apiClient.post<Tape>("/what_if/tape_check/record", {
        accessToken: accessToken ?? undefined,
        body: { ...shared, gateway_url: form.current, read_only_calls: form.calls ? JSON.parse(form.calls) : [] },
      }),
    );
  };
  const replay = async () =>
    setCheck(
      await apiClient.post<TapeCheck>("/what_if/tape_check/replay", {
        accessToken: accessToken ?? undefined,
        body: { ...shared, tape, gateway_url: form.candidate },
      }),
    );

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <Disc3 />
          {t("whatIf.tapeCheck.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.tapeCheck.description")}</PageHeaderDescription>
      </PageHeader>
      <Card>
        <CardContent className="grid grid-cols-3 gap-3">
          {field("current")}
          {field("key", true)}
          {field("header")}
          {field("credential", true)}
          <div className="col-span-2">{field("calls")}</div>
          <Button onClick={record}>{t("whatIf.tapeCheck.record")}</Button>
          {field("candidate")}
          <Button disabled={!tape} onClick={replay}>
            {t("whatIf.tapeCheck.replay")}
          </Button>
        </CardContent>
      </Card>
      {tape && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              {t("whatIf.tapeCheck.tape", { version: tape.version, count: tape.exchanges.length })}
              {check && (
                <Badge variant={check.changed ? "destructive" : "secondary"}>
                  {t("whatIf.tapeCheck.result", { version: check.candidate_version, count: check.changed })}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("whatIf.tapeCheck.exchange")}</TableHead>
                  <TableHead>{t("whatIf.tapeCheck.recorded")}</TableHead>
                  <TableHead>{t("whatIf.tapeCheck.replayed")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {tape.exchanges.map((exchange, index) => {
                  const replayed = check?.exchanges[index];
                  const tools = Object.keys(exchange.answer.tools);
                  return (
                    <TableRow key={exchange.name}>
                      <TableCell className="font-medium">{exchange.name}</TableCell>
                      <TableCell>
                        {exchange.status}
                        {tools.length > 0 && <code className="ml-2 text-xs">{tools.join(", ")}</code>}
                      </TableCell>
                      <TableCell>
                        {replayed?.verdict === "same" && (
                          <Badge variant="secondary">{t("whatIf.tapeCheck.same")}</Badge>
                        )}
                        {replayed?.verdict === "changed" && (
                          <div className="flex flex-wrap gap-1">
                            {replayed.changes.map((change) => (
                              <Badge key={change} variant="destructive">
                                {change}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </Page>
  );
}
