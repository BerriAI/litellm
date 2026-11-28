import { useState } from "react";
import { useTranslation } from "react-i18next";
import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { buildBulkKeyUpdate, type BulkKeyAction } from "./bulkKeyActions";

const actions: BulkKeyAction[] = ["budget", "rpm", "tpm", "expiry", "block", "unblock", "delete"];
type Outcome = { key: string; error: string | null };
interface Props {
  accessToken: string;
  teamId: string;
  selectedKeys: string[];
  onUpdated: () => void;
}

export default function BulkKeyActions({ accessToken, teamId, selectedKeys, onUpdated }: Props) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [targets, setTargets] = useState<string[]>([]);
  const [action, setAction] = useState<BulkKeyAction>("budget");
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [outcomes, setOutcomes] = useState<Outcome[] | null>(null);
  const [error, setError] = useState("");
  const needsValue = ["budget", "rpm", "tpm", "expiry"].includes(action);
  const update = buildBulkKeyUpdate(action, value);
  const invalid = needsValue && !update;

  const start = () => {
    setTargets([...selectedKeys]);
    setAction("budget");
    setValue("");
    setOutcomes(null);
    setError("");
    setOpen(true);
  };

  const changeAction = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setAction(event.target.value as BulkKeyAction);
    setValue("");
    setError("");
  };

  const apply = async (): Promise<Outcome[]> => {
    if (update && needsValue) {
      const response = await apiClient.post<components["schemas"]["BulkUpdateKeyResponse"]>("/team/key/bulk_update", {
        accessToken,
        body: { team_id: teamId, key_ids: targets, update_fields: update },
      });
      return targets.map((key) => {
        const failure = response.failed_updates.find((item) => item.key === key);
        if (failure) return { key, error: failure.failed_reason };
        const success = response.successful_updates.some((item) => item.key === key);
        return { key, error: success ? null : t("teams.bulkKeys.noResult") };
      });
    }
    if (action === "delete") {
      await apiClient.post("/key/delete", { accessToken, body: { keys: targets } });
      return targets.map((key) => ({ key, error: null }));
    }
    const results: Outcome[] = [];
    for (const key of targets) {
      try {
        await apiClient.post(action === "block" ? "/key/block" : "/key/unblock", { accessToken, body: { key } });
        results.push({ key, error: null });
      } catch (cause) {
        results.push({ key, error: cause instanceof Error ? cause.message : String(cause) });
      }
    }
    return results;
  };

  const submit = async () => {
    const cannotApply = invalid || targets.length === 0;
    if (busy || outcomes || cannotApply) return;
    setError("");
    setBusy(true);
    try {
      setOutcomes(await apply());
      onUpdated();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Button variant="outline" disabled={selectedKeys.length === 0 || selectedKeys.length > 500} onClick={start}>
        {t("teams.bulkKeys.title", { count: selectedKeys.length })}
      </Button>
      <Dialog
        open={open}
        onOpenChange={(next) => {
          if (!busy) setOpen(next);
        }}
        disablePointerDismissal={busy}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("teams.bulkKeys.title", { count: targets.length })}</DialogTitle>
          </DialogHeader>
          <Label htmlFor="bulk-key-action">{t("teams.bulkKeys.action")}</Label>
          <select id="bulk-key-action" value={action} disabled={busy || outcomes !== null} onChange={changeAction}>
            {actions.map((item) => (
              <option key={item} value={item}>
                {t(`teams.bulkKeys.${item}`)}
              </option>
            ))}
          </select>
          {needsValue && (
            <>
              <Label htmlFor="bulk-key-value">
                {t(action === "expiry" ? "teams.bulkKeys.duration" : "teams.bulkKeys.value")}
              </Label>
              <Input
                id="bulk-key-value"
                value={value}
                disabled={busy || outcomes !== null}
                onChange={(event) => setValue(event.target.value)}
              />
            </>
          )}
          <p>{t("teams.bulkKeys.summary", { action: t(`teams.bulkKeys.${action}`), count: targets.length })}</p>
          {action === "delete" && <p>{t("teams.bulkKeys.deleteWarning")}</p>}
          {invalid && <p role="status">{t("teams.bulkKeys.invalid")}</p>}
          {error && <p role="alert">{error}</p>}
          {outcomes && (
            <div aria-live="polite">
              <p>
                {t("teams.bulkKeys.result", {
                  success: outcomes.filter((item) => item.error === null).length,
                  total: outcomes.length,
                })}
              </p>
              <ul>
                {outcomes
                  .filter((item) => item.error !== null)
                  .map((item) => (
                    <li key={item.key}>
                      {item.key.slice(0, 8)}… {item.error}
                    </li>
                  ))}
              </ul>
            </div>
          )}
          <Button
            variant={action === "delete" ? "destructive" : "default"}
            disabled={busy || outcomes !== null || invalid}
            onClick={() => void submit()}
          >
            {busy ? t("teams.bulkKeys.applying") : t("teams.bulkKeys.confirm", { count: targets.length })}
          </Button>
        </DialogContent>
      </Dialog>
    </>
  );
}
