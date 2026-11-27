import { useState, type ChangeEvent } from "react";
import { useTranslation } from "react-i18next";
import { apiClient, type Member } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { memberImportBudget, parseMemberEmails, type MemberImportRow } from "./bulkMembers";

type Result = components["schemas"]["TeamMemberAddResult"];
interface Props {
  accessToken: string;
  teamId: string;
  members: Member[];
  onAdded: () => void;
}

export default function BulkAddMembers({ accessToken, teamId, members, onAdded }: Props) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [role, setRole] = useState("user");
  const [budget, setBudget] = useState("");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Result[] | null>(null);
  const [submittedRows, setSubmittedRows] = useState<MemberImportRow[]>([]);
  const [error, setError] = useState("");
  const rows = parseMemberEmails(
    text,
    members.flatMap((member) => (member.user_email ? [member.user_email] : [])),
  );
  const displayedRows = results ? submittedRows : rows;
  const ready = displayedRows.filter((row) => row.status === "ready");
  const maxBudget = memberImportBudget(budget);

  const invalidCount = ready.length === 0 || ready.length > 500;
  const invalidInput = invalidCount || maxBudget === "invalid";
  const submitDisabled = busy || invalidInput || results !== null;

  const submit = async () => {
    if (submitDisabled) return;
    setSubmittedRows(rows);
    setBusy(true);
    setError("");
    try {
      const response = await apiClient.post<components["schemas"]["BulkTeamMemberAddResponse"]>(
        "/team/bulk_member_add",
        {
          accessToken,
          body: {
            team_id: teamId,
            members: ready.map((row) => ({ user_email: row.email, role })),
            ...(maxBudget === undefined ? {} : { max_budget_in_team: maxBudget }),
          },
        },
      );
      setResults(response.results);
      onAdded();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  const start = () => {
    setText("");
    setRole("user");
    setBudget("");
    setResults(null);
    setError("");
    setOpen(true);
  };

  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      setText(await file.text());
      setError("");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setBusy(false);
    }
  };

  const rowLabel = (row: MemberImportRow) => {
    if (row.status !== "ready") return t(`teams.bulkMembers.${row.status}`);
    const result = results?.find((item) => item.user_email?.toLowerCase() === row.email.toLowerCase());
    if (!result) return t("teams.bulkMembers.ready");
    if (result.success) return t("teams.bulkMembers.added");
    return `${t("teams.bulkMembers.failed")} ${result.error ?? ""}`;
  };

  return (
    <>
      <Button variant="outline" onClick={start}>
        {t("teams.bulkMembers.title")}
      </Button>
      <Dialog
        open={open}
        onOpenChange={(value) => {
          if (!busy) setOpen(value);
        }}
        disablePointerDismissal={busy}
      >
        <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle>{t("teams.bulkMembers.title")}</DialogTitle>
          </DialogHeader>
          <Label htmlFor="bulk-member-emails">{t("teams.bulkMembers.emails")}</Label>
          <Textarea
            id="bulk-member-emails"
            value={text}
            disabled={busy || results !== null}
            onChange={(event) => setText(event.target.value)}
          />
          <Label htmlFor="bulk-member-file">{t("teams.bulkMembers.csv")}</Label>
          <Input
            id="bulk-member-file"
            type="file"
            accept=".csv,text/csv"
            disabled={busy || results !== null}
            onChange={(event) => void importFile(event)}
          />
          <Label htmlFor="bulk-member-role">{t("teams.bulkMembers.role")}</Label>
          <select
            id="bulk-member-role"
            value={role}
            disabled={busy || results !== null}
            onChange={(event) => setRole(event.target.value)}
          >
            <option value="user">{t("teams.bulkMembers.user")}</option>
            <option value="admin">{t("teams.bulkMembers.admin")}</option>
          </select>
          <Label htmlFor="bulk-member-budget">{t("teams.bulkMembers.budget")}</Label>
          <Input
            id="bulk-member-budget"
            type="number"
            min={0}
            step="any"
            value={budget}
            disabled={busy || results !== null}
            onChange={(event) => setBudget(event.target.value)}
          />
          {maxBudget === "invalid" && <p role="alert">{t("teams.bulkMembers.invalidBudget")}</p>}
          <p>{t("teams.bulkMembers.summary", { count: ready.length })}</p>
          {ready.length > 500 && <p role="alert">{t("teams.bulkMembers.limit")}</p>}
          {error && <p role="alert">{error}</p>}
          <ul aria-live="polite">
            {displayedRows.map((row, index) => (
              <li key={`${index}-${row.email}`}>
                {row.email} {rowLabel(row)}
              </li>
            ))}
          </ul>
          <Button onClick={() => void submit()} disabled={submitDisabled}>
            {busy ? t("teams.bulkMembers.adding") : t("teams.bulkMembers.confirm", { count: ready.length })}
          </Button>
        </DialogContent>
      </Dialog>
    </>
  );
}
