"use client";

import { useTranslation } from "react-i18next";
import React, { useEffect, useState } from "react";
import { PackageMinus } from "lucide-react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface LocalPatch {
  path: string;
  added: number;
  removed: number;
  first_change: string;
  found_in: string | null;
  verdict: string;
}
interface PatchDrop {
  tree: string;
  installed: string;
  checked_against: string[];
  patches: LocalPatch[];
  droppable: number;
}

export default function PatchDropPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [data, setData] = useState<PatchDrop | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    apiClient.get<PatchDrop>("/what_if/patch_drop", { accessToken }).then(setData);
  }, [accessToken]);

  const stats = [
    ["whatIf.patchDrop.installed", data?.installed ?? "-"],
    ["whatIf.patchDrop.carried", data?.patches.length ?? "-"],
    ["whatIf.patchDrop.droppable", data?.droppable ?? "-"],
  ] as const;

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <PackageMinus />
          {t("whatIf.patchDrop.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("whatIf.patchDrop.description")}</PageHeaderDescription>
      </PageHeader>
      <div className="grid grid-cols-3 gap-4">
        {stats.map(([label, value]) => (
          <Card key={label}>
            <CardHeader>
              <CardTitle className="text-sm text-muted-foreground">{t(label)}</CardTitle>
            </CardHeader>
            <CardContent className="text-2xl font-semibold">{value}</CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("whatIf.patchDrop.file")}</TableHead>
                <TableHead>{t("whatIf.patchDrop.lines")}</TableHead>
                <TableHead>{t("whatIf.patchDrop.foundIn")}</TableHead>
                <TableHead>{t("whatIf.patchDrop.verdict")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.patches.map((patch) => (
                <TableRow key={patch.path}>
                  <TableCell>
                    <div className="font-medium">{patch.path}</div>
                    <code className="text-xs text-muted-foreground">+ {patch.first_change.slice(0, 70)}</code>
                  </TableCell>
                  <TableCell>
                    +{patch.added} -{patch.removed}
                  </TableCell>
                  <TableCell>
                    <Badge variant={patch.found_in ? "secondary" : "destructive"}>
                      {patch.found_in?.replace("origin/", "") ?? t("whatIf.patchDrop.nowhere")}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-semibold">{patch.verdict}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </Page>
  );
}
