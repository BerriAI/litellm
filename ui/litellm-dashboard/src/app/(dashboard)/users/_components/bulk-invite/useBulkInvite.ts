import Papa from "papaparse";
import { useState } from "react";

import { useProxySettingsQuery } from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";
import { downloadBlob } from "@/components/EntityUsageExport/utils";
import { invitationCreateCall, userCreateCall } from "@/components/networking";
import { toast } from "@/lib/toast";

import { createUserFromRow, type BulkInviteApi, type CreateUserOutcome } from "./createUser";
import { parseCsvFile, type CsvParseResult, type CsvRow } from "./parseCsv";

export type BulkInviteRow = CsvRow & {
  status?: "pending" | CreateUserOutcome["status"];
  key?: string;
  invitation_link?: string;
};

type UploadProblem = { kind: "fileError" | "structureError" | "parseError"; message: string };

interface UseBulkInviteOptions {
  accessToken: string;
  teams: readonly { team_id: string }[] | null;
  onUsersCreated?: () => void;
}

const toResultRow = (row: BulkInviteRow) => ({
  user_email: row.user_email,
  user_role: row.user_role,
  status: row.status,
  key: row.key || "",
  invitation_link: row.invitation_link || "",
  error: row.error || "",
});

export const useBulkInvite = ({ accessToken, teams, onUsersCreated }: UseBulkInviteOptions) => {
  const ssoEnabled = Boolean(useProxySettingsQuery(accessToken).data?.SSO_ENABLED);
  const [rows, setRows] = useState<BulkInviteRow[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [problem, setProblem] = useState<UploadProblem | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pageIndex, setPageIndex] = useState(0);

  const applyParseResult = (result: CsvParseResult) => {
    switch (result.kind) {
      case "fileError":
        setProblem({ kind: result.kind, message: result.message });
        if (result.toast) toast.fromError(result.toast);
        return;
      case "structureError":
      case "parseError":
        setRows([]);
        setProblem({ kind: result.kind, message: result.message });
        return;
      case "rows":
        setRows(result.rows);
        if (result.rowError) setProblem({ kind: "parseError", message: result.rowError });
        else toast.success(`Successfully parsed ${result.rows.length} users`);
    }
  };

  const uploadFile = async (file: File) => {
    setProblem(null);
    setSelectedFile(file);
    applyParseResult(await parseCsvFile(file, teams?.map((team) => team.team_id) ?? []));
  };

  const removeFile = () => {
    setSelectedFile(null);
    setRows([]);
    setProblem(null);
  };

  const backToUpload = () => {
    setRows([]);
    setProblem((current) => (current?.kind === "parseError" ? null : current));
    setPageIndex(0);
  };

  const createUsers = async () => {
    setIsProcessing(true);
    const pending = rows.map((row): BulkInviteRow => ({ ...row, status: "pending" }));
    setRows(pending);

    const api: BulkInviteApi = {
      createUser: (request) => userCreateCall(accessToken, null, request),
      createInvitation: (userId) => invitationCreateCall(accessToken, userId),
    };
    const settings = { ssoEnabled, baseUrl: window.location.origin };
    let anySuccessful = false;
    for (const [index, row] of pending.entries()) {
      const outcome = await createUserFromRow(row, api, settings);
      anySuccessful ||= outcome.status === "success";
      setRows((current) => current.map((existing, i) => (i === index ? { ...existing, ...outcome } : existing)));
    }

    setIsProcessing(false);
    if (anySuccessful) onUsersCreated?.();
  };

  const downloadResults = () =>
    downloadBlob(new Blob([Papa.unparse(rows.map(toResultRow))], { type: "text/csv" }), "bulk_users_results.csv");

  return {
    rows,
    selectedFile,
    fileError: problem?.kind === "fileError" ? problem.message : null,
    csvStructureError: problem?.kind === "structureError" ? problem.message : null,
    parseError: problem?.kind === "parseError" ? problem.message : null,
    isProcessing,
    pageIndex,
    setPageIndex,
    uploadFile,
    removeFile,
    backToUpload,
    createUsers,
    downloadResults,
  };
};
