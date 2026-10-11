import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Plus, Trash2, Send, ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "@/lib/toast";
import { SimpleTooltip } from "@/components/ui/tooltip";
import {
  getSpendReportEmailSettings,
  getSpendReportEmailStatus,
  updateSpendReportEmailSettings,
  sendSpendReportEmailTest,
  teamListCall,
  type SpendReportEmailAlertItem,
  type SpendReportEmailSettingsValues,
  type SpendReportEmailStatusResponse,
} from "./networking";
import type { Team } from "@/components/key_team_helpers/key_list";

interface SpendReportEmailSettingsProps {
  accessToken: string | null;
  teams?: Team[];
  defaultExpanded?: boolean;
}

type GroupByOptionId = "team" | "team_key" | "team_model" | "team_key_model";

interface GroupByOption {
  readonly id: GroupByOptionId;
  readonly label: string;
  readonly description: string;
}

const GROUP_BY_OPTIONS: readonly GroupByOption[] = [
  { id: "team", label: "Team", description: "Aggregates spend grouped by team" },
  { id: "team_key", label: "Team + Keys", description: "Aggregates spend grouped by team and API key" },
  { id: "team_model", label: "Team + Models", description: "Aggregates spend grouped by team and model" },
  { id: "team_key_model", label: "Team + Key + Models", description: "Aggregates spend grouped by team, key, and model" },
];

const FREQUENCY_LABELS: Record<"daily" | "monthly", string> = {
  daily: "Daily (Previous Day)",
  monthly: "Monthly (1st of Month)",
};

const buildInitialAlerts = (values: SpendReportEmailSettingsValues): SpendReportEmailAlertItem[] => {
  if (values.alerts && values.alerts.length > 0) {
    return values.alerts;
  }

  if (values.frequency === "both") {
    return [
      {
        id: "daily-alert",
        name: "Daily Spend Report",
        enabled: values.enabled,
        frequency: "daily",
        send_time: values.daily_send_time || "09:00",
        recipient_emails: values.recipient_emails || [],
        group_by: values.group_by || ["team"],
        team_id: null,
      },
      {
        id: "monthly-alert",
        name: "Monthly Spend Report",
        enabled: values.enabled,
        frequency: "monthly",
        send_time: values.monthly_send_time || "09:00",
        recipient_emails: values.recipient_emails || [],
        group_by: values.group_by || ["team"],
        team_id: null,
      },
    ];
  }

  return [
    {
      id: "spend-alert-1",
      name: "Spend Report Alert",
      enabled: values.enabled ?? false,
      frequency: values.frequency === "monthly" ? "monthly" : "daily",
      send_time:
        values.frequency === "monthly"
          ? values.monthly_send_time || "09:00"
          : values.daily_send_time || "09:00",
      recipient_emails: values.recipient_emails || [],
      group_by: values.group_by || ["team"],
      team_id: null,
    },
  ];
};

export const SpendReportEmailSettings: React.FC<SpendReportEmailSettingsProps> = ({
  accessToken,
  teams: initialTeams,
  defaultExpanded,
}) => {
  const [loading, setLoading] = useState<boolean>(Boolean(accessToken));
  const [saving, setSaving] = useState(false);
  const [testingAlertId, setTestingAlertId] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState<boolean>(defaultExpanded ?? false);
  const [emailStatus, setEmailStatus] = useState<SpendReportEmailStatusResponse | null>(null);

  const [alerts, setAlerts] = useState<SpendReportEmailAlertItem[]>([]);
  const [recipientsInputs, setRecipientsInputs] = useState<Record<string, string>>({});
  const [availableTeams, setAvailableTeams] = useState<Team[]>([]);

  useEffect(() => {
    if (!accessToken) return;
    getSpendReportEmailStatus(accessToken)
      .then((status) => setEmailStatus(status))
      .catch(() => setEmailStatus({ configured: false, provider: null }));
  }, [accessToken]);

  useEffect(() => {
    let active = true;
    if (!accessToken) {
      return;
    }

    getSpendReportEmailSettings(accessToken)
      .then((response) => {
        if (active && response?.values) {
          const loadedAlerts = buildInitialAlerts(response.values);
          setAlerts(loadedAlerts);
          const initialInputs: Record<string, string> = {};
          loadedAlerts.forEach((a) => {
            initialInputs[a.id] = (a.recipient_emails || []).join(", ");
          });
          setRecipientsInputs(initialInputs);
        }
      })
      .catch((error) => {
        if (active) {
          console.error("Failed to load spend report email settings:", error);
          toast.fromError("Failed to load spend report email settings");
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [accessToken]);

  useEffect(() => {
    if (initialTeams && initialTeams.length > 0) {
      return;
    }
    if (!accessToken) {
      return;
    }

    let active = true;
    teamListCall(accessToken, null)
      .then((response: unknown) => {
        if (!active) return;
        const resObj = response as { teams?: Team[]; data?: Team[] } | Team[] | null;
        const list = Array.isArray(resObj)
          ? resObj
          : Array.isArray(resObj?.teams)
            ? resObj.teams
            : Array.isArray(resObj?.data)
              ? resObj.data
              : [];
        setAvailableTeams(list);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [accessToken, initialTeams]);

  const teams = initialTeams && initialTeams.length > 0 ? initialTeams : availableTeams;

  const handleAddAlert = () => {
    setIsExpanded(true);
    const newId =
      typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `alert-${Date.now()}`;
    const newAlert: SpendReportEmailAlertItem = {
      id: newId,
      name: `Spend Report Alert ${alerts.length + 1}`,
      enabled: true,
      frequency: "daily",
      send_time: "09:00",
      recipient_emails: [],
      group_by: ["team"],
      team_id: null,
    };
    setAlerts((prev) => [...prev, newAlert]);
    setRecipientsInputs((prev) => ({ ...prev, [newId]: "" }));
  };

  const handleDeleteAlert = (alertId: string) => {
    setAlerts((prev) => prev.filter((a) => a.id !== alertId));
    setRecipientsInputs((prev) => {
      const copy = { ...prev };
      delete copy[alertId];
      return copy;
    });
  };

  const handleUpdateAlert = (alertId: string, updates: Partial<SpendReportEmailAlertItem>) => {
    setAlerts((prev) => prev.map((a) => (a.id === alertId ? { ...a, ...updates } : a)));
  };

  const handleRecipientsChange = (alertId: string, value: string) => {
    setRecipientsInputs((prev) => ({ ...prev, [alertId]: value }));
  };

  const handleGroupByToggle = (alertId: string, optionId: GroupByOptionId, checked: boolean) => {
    setAlerts((prev) =>
      prev.map((a) => {
        if (a.id !== alertId) return a;
        let updated: GroupByOptionId[];
        if (checked) {
          updated = [...a.group_by, optionId];
        } else {
          updated = a.group_by.filter((g) => g !== optionId);
        }
        if (updated.length === 0) {
          updated = ["team"];
        }
        return { ...a, group_by: updated };
      }),
    );
  };

  const handleSave = async () => {
    if (!accessToken) return;

    setSaving(true);
    try {
      const parsedAlerts: SpendReportEmailAlertItem[] = alerts.map((a) => {
        const raw = recipientsInputs[a.id] ?? (a.recipient_emails || []).join(", ");
        const parsed = raw
          .split(",")
          .map((e) => e.trim())
          .filter((e) => e.length > 0);
        return {
          ...a,
          recipient_emails: parsed,
        };
      });

      const anyEnabled = parsedAlerts.some((a) => a.enabled);
      const dailyAlert = parsedAlerts.find((a) => a.frequency === "daily");
      const monthlyAlert = parsedAlerts.find((a) => a.frequency === "monthly");
      const allRecipients = Array.from(new Set(parsedAlerts.flatMap((a) => a.recipient_emails)));

      const payload: SpendReportEmailSettingsValues = {
        enabled: anyEnabled,
        frequency: dailyAlert && monthlyAlert ? "both" : monthlyAlert ? "monthly" : "daily",
        daily_send_time: dailyAlert?.send_time || "09:00",
        monthly_send_time: monthlyAlert?.send_time || "09:00",
        recipient_emails: allRecipients,
        group_by: parsedAlerts[0]?.group_by || ["team"],
        alerts: parsedAlerts,
      };

      await updateSpendReportEmailSettings(accessToken, payload);
      setAlerts(parsedAlerts);
      toast.success("Spend report email settings saved successfully");
    } catch (error) {
      console.error("Failed to save spend report email settings:", error);
      toast.fromError("Failed to save spend report email settings");
    } finally {
      setSaving(false);
    }
  };

  const handleSendTestReport = async (alert: SpendReportEmailAlertItem) => {
    if (!accessToken) return;

    const raw = recipientsInputs[alert.id] ?? (alert.recipient_emails || []).join(", ");
    const parsedRecipients = raw
      .split(",")
      .map((e) => e.trim())
      .filter((e) => e.length > 0);

    if (parsedRecipients.length === 0) {
      toast.error("Please provide at least one recipient email address to send a test report");
      return;
    }

    setTestingAlertId(alert.id);
    try {
      const result = await sendSpendReportEmailTest(accessToken, {
        frequency: alert.frequency,
        recipient_emails: parsedRecipients,
        group_by: alert.group_by,
        team_id: alert.team_id ?? null,
      });
      toast.success(
        `Test spend report email sent! Total spend: $${result.total_spend ?? 0} across ${result.total_requests ?? 0} request(s).`,
      );
    } catch (error) {
      console.error("Failed to send test spend report email:", error);
      toast.fromError(error);
    } finally {
      setTestingAlertId(null);
    }
  };

  const activeAlertsCount = alerts.filter((a) => a.enabled).length;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-base font-semibold">Scheduled Spend Reports (Email / SMTP)</CardTitle>
              <Badge variant={activeAlertsCount > 0 ? "default" : "outline"}>
                {activeAlertsCount > 0 ? `${activeAlertsCount} Active` : "Disabled"}
              </Badge>
            </div>
            <CardDescription className="text-sm text-muted-foreground">
              Automatically send daily and/or monthly spend reports to configured email addresses. Includes cumulative
              breakdowns by team, API keys, and models.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2 self-start sm:self-auto shrink-0">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleAddAlert}
              disabled={loading || saving}
              className="flex items-center gap-1.5"
            >
              <Plus className="h-4 w-4" />
              Add Alert
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded((prev) => !prev)}
              aria-label={isExpanded ? "Collapse spend report settings" : "Expand spend report settings"}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              <span>{isExpanded ? "Collapse" : "Expand"}</span>
              {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent>
        <Separator className="mb-6" />

        {loading ? (
          <div className="space-y-4">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-28 w-full" />
            <Skeleton className="h-28 w-full" />
          </div>
        ) : alerts.length === 0 ? (
          <div className="text-center py-12 border border-dashed rounded-lg space-y-3">
            <p className="text-sm font-medium text-foreground">No scheduled spend report alerts configured</p>
            <p className="text-xs text-muted-foreground max-w-md mx-auto">
              Configure daily or monthly alerts for your entire proxy or specific teams to keep stakeholders informed of
              LLM spend.
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleAddAlert}
              className="mt-2 flex items-center gap-1.5 mx-auto"
            >
              <Plus className="h-4 w-4" />
              Add First Alert
            </Button>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="space-y-4">
              {alerts.map((alert, index) => {
                const recipientsValue = recipientsInputs[alert.id] ?? (alert.recipient_emails || []).join(", ");
                const isTesting = testingAlertId === alert.id;
                const selectedTeamLabel = alert.team_id
                  ? teams.find((t) => t.team_id === alert.team_id)?.team_alias || alert.team_id
                  : "All Teams (No filter)";

                return (
                  <Card key={alert.id} className="border border-border bg-card shadow-sm">
                    <CardHeader className="pb-3 pt-4 px-4 sm:px-6">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                        <div className="flex items-center gap-3 flex-1 min-w-0">
                          <Input
                            value={alert.name}
                            onChange={(e) => handleUpdateAlert(alert.id, { name: e.target.value })}
                            placeholder={`Alert ${index + 1}`}
                            className="font-semibold text-sm max-w-xs h-8"
                          />
                          <Badge variant="secondary" className="capitalize text-xs shrink-0">
                            {alert.frequency}
                          </Badge>
                          {alert.team_id ? (
                            <Badge variant="outline" className="text-xs shrink-0">
                              Team: {teams.find((t) => t.team_id === alert.team_id)?.team_alias || alert.team_id}
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="text-xs shrink-0">
                              All Teams
                            </Badge>
                          )}
                        </div>

                        <div className="flex items-center gap-3 shrink-0">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-muted-foreground">{alert.enabled ? "Active" : "Disabled"}</span>
                            <Switch
                              checked={alert.enabled}
                              onCheckedChange={(checked) => handleUpdateAlert(alert.id, { enabled: checked })}
                              disabled={saving}
                            />
                          </div>

                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                            onClick={() => handleDeleteAlert(alert.id)}
                            disabled={saving}
                            title="Delete alert"
                            aria-label={`Delete alert ${alert.name}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardHeader>

                    <CardContent className="px-4 sm:px-6 pb-4 pt-0 space-y-4">
                      <Separator className="mb-4" />

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="space-y-1.5">
                          <label className="text-xs font-medium text-foreground">Frequency</label>
                          <Select
                            value={alert.frequency}
                            onValueChange={(val: "daily" | "monthly" | null) => {
                              if (val) handleUpdateAlert(alert.id, { frequency: val });
                            }}
                          >
                            <SelectTrigger className="w-full h-9 text-xs">
                              <SelectValue placeholder="Select frequency">
                                {FREQUENCY_LABELS[alert.frequency]}
                              </SelectValue>
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="daily">Daily (Previous Day)</SelectItem>
                              <SelectItem value="monthly">Monthly (1st of Month)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-1.5">
                          <label className="text-xs font-medium text-foreground">Send Time (UTC)</label>
                          <Input
                            placeholder="09:00"
                            value={alert.send_time}
                            onChange={(e) => handleUpdateAlert(alert.id, { send_time: e.target.value })}
                            className="h-9 text-xs"
                          />
                        </div>

                        <div className="space-y-1.5">
                          <label className="text-xs font-medium text-foreground">Team Scope</label>
                          <Select
                            value={alert.team_id || "all"}
                            onValueChange={(val: string | null) => {
                              handleUpdateAlert(alert.id, { team_id: !val || val === "all" ? null : val });
                            }}
                          >
                            <SelectTrigger className="w-full h-9 text-xs">
                              <SelectValue placeholder="Select team">
                                {selectedTeamLabel}
                              </SelectValue>
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="all">All Teams (No filter)</SelectItem>
                              {teams.map((t) => (
                                <SelectItem key={t.team_id} value={t.team_id}>
                                  {t.team_alias || t.team_id}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <div className="space-y-1.5">
                        <label className="text-xs font-medium text-foreground">Recipient Email Addresses</label>
                        <Input
                          placeholder="finops@example.com, manager@example.com"
                          value={recipientsValue}
                          onChange={(e) => handleRecipientsChange(alert.id, e.target.value)}
                          className="h-9 text-xs"
                        />
                        <p className="text-[11px] text-muted-foreground">
                          Comma-separated list of target emails to receive this alert.
                        </p>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-medium text-foreground">Cumulative Report Groupings</label>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                          {GROUP_BY_OPTIONS.map((opt) => {
                            const isChecked = alert.group_by.includes(opt.id);
                            const elementId = `groupby-${alert.id}-${opt.id}`;
                            return (
                              <div
                                key={opt.id}
                                className="flex items-start space-x-2 p-2 rounded-md border border-border bg-muted/30"
                              >
                                <Checkbox
                                  id={elementId}
                                  checked={isChecked}
                                  onCheckedChange={(checked) =>
                                    handleGroupByToggle(alert.id, opt.id, Boolean(checked))
                                  }
                                  className="mt-0.5"
                                />
                                <label htmlFor={elementId} className="cursor-pointer space-y-0.5 select-none">
                                  <div className="text-xs font-medium text-foreground">{opt.label}</div>
                                  <div className="text-[10px] text-muted-foreground leading-tight">
                                    {opt.description}
                                  </div>
                                </label>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      <div className="flex justify-end pt-2">
                        <SimpleTooltip
                          content={
                            emailStatus && !emailStatus.configured
                              ? "Email delivery is not configured. Configure SMTP server settings above or set SENDGRID_API_KEY environment variable."
                              : undefined
                          }
                        >
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => handleSendTestReport(alert)}
                            disabled={isTesting || saving || (emailStatus !== null && !emailStatus.configured)}
                            className="flex items-center gap-1.5 text-xs h-8"
                          >
                            <Send className="h-3.5 w-3.5" />
                            {isTesting ? "Sending Test..." : "Send Test Report Now"}
                          </Button>
                        </SimpleTooltip>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-border">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleAddAlert}
                disabled={loading || saving}
                className="flex items-center gap-1.5"
              >
                <Plus className="h-4 w-4" />
                Add Another Alert
              </Button>

              <Button onClick={handleSave} disabled={saving || loading}>
                {saving ? "Saving..." : "Save Spend Report Settings"}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
      )}
    </Card>
  );
};

export default SpendReportEmailSettings;
