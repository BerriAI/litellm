import React, { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { toast } from "@/lib/toast";
import { ExternalLink, Loader2, Mail } from "lucide-react";
import Link from "next/link";
import {
  formatDate,
  getSpendReportEmailStatus,
  sendSpendReportEmailTest,
  type SpendReportEmailStatusResponse,
} from "./networking";
import type { Team } from "./key_team_helpers/key_list";

export interface EmailSpendSummaryModalProps {
  isOpen: boolean;
  onClose: () => void;
  accessToken: string | null;
  startDate?: string | null;
  endDate?: string | null;
  teams?: Team[] | null;
  initialTeamId?: string | null;
  emailStatus?: SpendReportEmailStatusResponse | null;
}

type GroupByOptionId = "team" | "team_key" | "team_model" | "team_key_model";

interface GroupByOption {
  readonly id: GroupByOptionId;
  readonly label: string;
  readonly description: string;
}

const GROUP_BY_OPTIONS: readonly GroupByOption[] = [
  { id: "team", label: "Team", description: "Aggregates spend by team" },
  { id: "team_key", label: "Team + Keys", description: "Aggregates spend by team and API key" },
  { id: "team_model", label: "Team + Models", description: "Aggregates spend by team and model" },
  { id: "team_key_model", label: "Team + Key + Models", description: "Aggregates spend by team, key, and model" },
];

const DEFAULT_GROUP_BY: GroupByOptionId[] = ["team"];

export const EmailSpendSummaryModal: React.FC<EmailSpendSummaryModalProps> = ({
  isOpen,
  onClose,
  accessToken,
  startDate,
  endDate,
  teams,
  initialTeamId,
  emailStatus,
}) => {
  const [recipientsInput, setRecipientsInput] = useState("");
  const [selectedTeamId, setSelectedTeamId] = useState<string>(initialTeamId || "all");
  const [groupBy, setGroupBy] = useState<GroupByOptionId[]>(DEFAULT_GROUP_BY);
  const [isSending, setIsSending] = useState(false);
  const [internalEmailStatus, setInternalEmailStatus] = useState<SpendReportEmailStatusResponse | null>(null);

  useEffect(() => {
    if (emailStatus !== undefined) {
      return;
    }
    if (!isOpen || !accessToken) return;
    let cancelled = false;
    getSpendReportEmailStatus(accessToken)
      .then((status) => {
        if (!cancelled) {
          setInternalEmailStatus(status);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setInternalEmailStatus(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, accessToken, emailStatus]);

  const effectiveEmailStatus = emailStatus !== undefined ? emailStatus : internalEmailStatus;
  const isEmailNotConfigured = effectiveEmailStatus !== null && !effectiveEmailStatus.configured;

  const fallbackStartDate = useMemo(() => {
    const d = new Date();
    d.setDate(d.getDate() - 7);
    return formatDate(d);
  }, []);
  const fallbackEndDate = useMemo(() => formatDate(new Date()), []);

  const resolvedStartDate = startDate || fallbackStartDate;
  const resolvedEndDate = endDate || fallbackEndDate;

  const handleGroupByToggle = (optionId: GroupByOptionId, checked: boolean) => {
    setGroupBy((prev) => {
      let updated: GroupByOptionId[];
      if (checked) {
        updated = [...prev, optionId];
      } else {
        updated = prev.filter((g) => g !== optionId);
      }
      if (updated.length === 0) {
        updated = ["team"];
      }
      return updated;
    });
  };

  const handleSend = async () => {
    if (!accessToken) {
      toast.error("Not authenticated");
      return;
    }

    const parsedRecipients = recipientsInput
      .split(",")
      .map((e) => e.trim())
      .filter((e) => e.length > 0);

    if (parsedRecipients.length === 0) {
      toast.error("Please enter at least one recipient email address");
      return;
    }

    setIsSending(true);
    try {
      const teamFilter = selectedTeamId !== "all" ? selectedTeamId : undefined;
      const sendPayload = {
        start_date: resolvedStartDate,
        end_date: resolvedEndDate,
        recipient_emails: parsedRecipients,
        group_by: groupBy,
        team_id: teamFilter,
        frequency: "daily" as const,
      };
      const result = await sendSpendReportEmailTest(accessToken, sendPayload);

      toast.success(
        `Spend summary email sent to ${parsedRecipients.length} recipient(s)! Total spend: $${result.total_spend ?? 0} across ${result.total_requests ?? 0} request(s).`,
      );
      onClose();
    } catch (error) {
      console.error("Failed to send spend summary email:", error);
      toast.fromError(error);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open && !isSending) onClose();
      }}
    >
      <DialogContent className="sm:max-w-[540px]">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <Mail className="size-5 text-primary" />
            <DialogTitle className="text-base font-semibold">Email Spend Summary</DialogTitle>
          </div>
          <DialogDescription className="text-xs text-muted-foreground">
            Send an on-demand spend report email for the selected period to target email addresses.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="p-3 rounded-lg border border-border bg-muted/40 flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Reporting Period:</span>
            <span className="font-medium text-foreground">
              {resolvedStartDate} to {resolvedEndDate}
            </span>
          </div>

          <div className="p-3 rounded-lg border border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-950/20 text-xs text-muted-foreground flex items-start gap-2">
            <div className="flex-1">
              Want to receive spend summaries on a schedule? You can configure automated daily, monthly, or
              team-specific email alerts.
              <div className="mt-1">
                <Link
                  href="/logging-and-alerts?tab=email-alerts"
                  className="text-primary hover:underline font-medium inline-flex items-center gap-1"
                  onClick={onClose}
                >
                  Configure Scheduled Email Alerts <ExternalLink className="size-3" />
                </Link>
              </div>
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-foreground">
              Target Email Address(es) <span className="text-destructive">*</span>
            </label>
            <Input
              placeholder="finance@example.com, ops@example.com"
              value={recipientsInput}
              onChange={(e) => setRecipientsInput(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">Enter one or more email addresses separated by commas.</p>
          </div>

          {teams && teams.length > 0 && (
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-foreground">Team Scope</label>
              <Select value={selectedTeamId} onValueChange={(val) => setSelectedTeamId(val || "all")}>
                <SelectTrigger>
                  <SelectValue placeholder="All Teams">
                    {selectedTeamId !== "all"
                      ? teams.find((t) => t.team_id === selectedTeamId)?.team_alias || selectedTeamId
                      : "All Teams"}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Teams</SelectItem>
                  {teams.map((t) => (
                    <SelectItem key={t.team_id} value={t.team_id}>
                      {t.team_alias || t.team_id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Cumulative Groupings</label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {GROUP_BY_OPTIONS.map((opt) => {
                const isChecked = groupBy.includes(opt.id);
                return (
                  <div
                    key={opt.id}
                    className="flex items-start space-x-2.5 p-2 rounded-md border border-border bg-muted/30"
                  >
                    <Checkbox
                      id={`summary-groupby-${opt.id}`}
                      checked={isChecked}
                      onCheckedChange={(checked) => handleGroupByToggle(opt.id, Boolean(checked))}
                      className="mt-0.5"
                    />
                    <label htmlFor={`summary-groupby-${opt.id}`} className="cursor-pointer space-y-0.5 select-none">
                      <div className="text-xs font-medium text-foreground">{opt.label}</div>
                      <div className="text-[11px] text-muted-foreground">{opt.description}</div>
                    </label>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="flex items-center justify-end gap-2 pt-3 border-t">
            <Button variant="outline" onClick={onClose} disabled={isSending}>
              Cancel
            </Button>
            <SimpleTooltip
              content={
                isEmailNotConfigured
                  ? "Email delivery is not configured. Configure SMTP server settings in Alerts settings or set SENDGRID_API_KEY environment variable."
                  : undefined
              }
            >
              <Button onClick={handleSend} disabled={isSending || isEmailNotConfigured}>
                {isSending && <Loader2 className="animate-spin size-4" />}
                {isSending ? "Sending Email..." : "Send Email Summary"}
              </Button>
            </SimpleTooltip>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default EmailSpendSummaryModal;
