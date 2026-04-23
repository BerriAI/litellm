import React, { useState, useEffect, useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Edit, Save, X } from "lucide-react";
import { getDefaultTeamSettings, updateDefaultTeamSettings } from "./networking";
import BudgetDurationDropdown, {
  getBudgetDurationLabel,
} from "./common_components/budget_duration_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import NotificationsManager from "./molecules/notifications_manager";
import { ModelSelect } from "./ModelSelect/ModelSelect";

interface TeamSSOSettingsProps {
  accessToken: string | null;
  userID: string;
  userRole: string;
}

const PERMISSION_OPTIONS = [
  "/key/generate",
  "/key/update",
  "/key/delete",
  "/key/regenerate",
  "/key/service-account/generate",
  "/key/{key_id}/regenerate",
  "/key/block",
  "/key/unblock",
  "/key/bulk_update",
  "/key/{key_id}/reset_spend",
  "/key/info",
  "/key/list",
  "/key/aliases",
  "/team/daily/activity",
];

interface SettingRowProps {
  label: string;
  description: string;
  isEditing: boolean;
  viewContent: React.ReactNode;
  editContent: React.ReactNode;
}

const SettingRow: React.FC<SettingRowProps> = ({
  label,
  description,
  isEditing,
  viewContent,
  editContent,
}) => (
  <div className="grid grid-cols-3 py-5 border-b border-border last:border-0">
    <div className="col-span-1 pr-6">
      <div className="text-sm font-semibold text-foreground">{label}</div>
      <div className="text-xs text-muted-foreground mt-1 leading-relaxed">
        {description}
      </div>
    </div>
    <div className="col-span-2 flex items-center">
      <div className="w-full">{isEditing ? editContent : viewContent}</div>
    </div>
  </div>
);

const NotSet = () => (
  <span className="text-muted-foreground italic">Not set</span>
);

const renderTags = (values: string[], displayFn?: (v: string) => string) => {
  if (!values || values.length === 0) return <NotSet />;
  return (
    <div className="flex flex-wrap gap-2">
      {values.map((v) => (
        <Badge key={v} variant="default">
          {displayFn ? displayFn(v) : v}
        </Badge>
      ))}
    </div>
  );
};

interface SettingsValues {
  max_budget: number | null;
  budget_duration: string | null;
  tpm_limit: number | null;
  rpm_limit: number | null;
  models: string[];
  team_member_permissions: string[];
}

const DEFAULT_VALUES: SettingsValues = {
  max_budget: null,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  models: [],
  team_member_permissions: [],
};

const TeamSSOSettings: React.FC<TeamSSOSettingsProps> = ({ accessToken }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [values, setValues] = useState<SettingsValues>(DEFAULT_VALUES);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editedValues, setEditedValues] =
    useState<SettingsValues>(DEFAULT_VALUES);
  const [saving, setSaving] = useState<boolean>(false);
  const [fetchError, setFetchError] = useState<boolean>(false);

  useEffect(() => {
    const fetchSettings = async () => {
      if (!accessToken) {
        setLoading(false);
        return;
      }
      try {
        const data = await getDefaultTeamSettings(accessToken);
        const fetched = { ...DEFAULT_VALUES, ...(data.values || {}) };
        setValues(fetched);
        setEditedValues(fetched);
      } catch (error) {
        console.error("Error fetching team SSO settings:", error);
        setFetchError(true);
        NotificationsManager.fromBackend("Failed to fetch team settings");
      } finally {
        setLoading(false);
      }
    };
    fetchSettings();
  }, [accessToken]);

  const handleSave = async () => {
    if (!accessToken) return;
    setSaving(true);
    try {
      const updatedSettings = await updateDefaultTeamSettings(
        accessToken,
        editedValues,
      );
      const newValues = {
        ...DEFAULT_VALUES,
        ...(updatedSettings.settings || {}),
      };
      setValues(newValues);
      setEditedValues(newValues);
      setIsEditing(false);
      NotificationsManager.success(
        "Default team settings updated successfully",
      );
    } catch (error) {
      console.error("Error updating team settings:", error);
      NotificationsManager.fromBackend("Failed to update team settings");
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditedValues(values);
  };

  const update = <K extends keyof SettingsValues>(
    key: K,
    value: SettingsValues[K],
  ) => {
    setEditedValues((prev) => ({ ...prev, [key]: value }));
  };

  const remainingPermissions = useMemo(
    () =>
      PERMISSION_OPTIONS.filter(
        (p) => !(editedValues.team_member_permissions ?? []).includes(p),
      ),
    [editedValues.team_member_permissions],
  );

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Skeleton className="h-32 w-full max-w-3xl" />
      </div>
    );
  }

  if (fetchError) {
    return (
      <Card className="p-6">
        <p className="text-sm">
          No team settings available or you do not have permission to view them.
        </p>
      </Card>
    );
  }

  return (
    <Card className="p-8">
      <div className="flex justify-between items-start mb-2">
        <div>
          <h3 className="text-xl font-semibold m-0 text-foreground">
            Default Team Settings
          </h3>
          <p className="text-muted-foreground text-sm mt-1">
            These settings will be applied by default when creating new teams.
          </p>
        </div>
        <div>
          {isEditing ? (
            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={handleCancel}
                disabled={saving}
              >
                Cancel
              </Button>
              <Button onClick={handleSave} disabled={saving}>
                <Save className="h-4 w-4" />
                {saving ? "Saving…" : "Save Changes"}
              </Button>
            </div>
          ) : (
            <Button variant="outline" onClick={() => setIsEditing(true)}>
              <Edit className="h-4 w-4" />
              Edit Settings
            </Button>
          )}
        </div>
      </div>

      <div className="mt-8">
        {/* Budget & Rate Limits */}
        <div className="mb-8">
          <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-2">
            Budget &amp; Rate Limits
          </div>
          <div className="border-t border-border">
            <SettingRow
              label="Max Budget"
              description="Maximum budget (in USD) for new automatically created teams."
              isEditing={isEditing}
              viewContent={
                values.max_budget != null ? (
                  <span>${Number(values.max_budget).toLocaleString()}</span>
                ) : (
                  <NotSet />
                )
              }
              editContent={
                <div className="relative max-w-xs">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">
                    $
                  </span>
                  <Input
                    type="number"
                    min={0}
                    step={0.01}
                    value={editedValues.max_budget ?? ""}
                    onChange={(e) =>
                      update(
                        "max_budget",
                        e.target.value === "" ? null : Number(e.target.value),
                      )
                    }
                    placeholder="Not set"
                    className="pl-6"
                  />
                </div>
              }
            />

            <SettingRow
              label="Budget Duration"
              description="How frequently the team's budget resets."
              isEditing={isEditing}
              viewContent={
                values.budget_duration ? (
                  <span>{getBudgetDurationLabel(values.budget_duration)}</span>
                ) : (
                  <NotSet />
                )
              }
              editContent={
                <div className="max-w-xs">
                  <BudgetDurationDropdown
                    value={editedValues.budget_duration}
                    onChange={(v) => update("budget_duration", v)}
                  />
                </div>
              }
            />

            <SettingRow
              label="TPM Limit"
              description="Maximum tokens per minute allowed across all models."
              isEditing={isEditing}
              viewContent={
                values.tpm_limit != null ? (
                  <span>{values.tpm_limit.toLocaleString()}</span>
                ) : (
                  <NotSet />
                )
              }
              editContent={
                <Input
                  type="number"
                  min={0}
                  className="max-w-xs"
                  value={editedValues.tpm_limit ?? ""}
                  onChange={(e) =>
                    update(
                      "tpm_limit",
                      e.target.value === "" ? null : Number(e.target.value),
                    )
                  }
                  placeholder="Not set"
                />
              }
            />

            <SettingRow
              label="RPM Limit"
              description="Maximum requests per minute allowed across all models."
              isEditing={isEditing}
              viewContent={
                values.rpm_limit != null ? (
                  <span>{values.rpm_limit.toLocaleString()}</span>
                ) : (
                  <NotSet />
                )
              }
              editContent={
                <Input
                  type="number"
                  min={0}
                  className="max-w-xs"
                  value={editedValues.rpm_limit ?? ""}
                  onChange={(e) =>
                    update(
                      "rpm_limit",
                      e.target.value === "" ? null : Number(e.target.value),
                    )
                  }
                  placeholder="Not set"
                />
              }
            />
          </div>
        </div>

        {/* Access & Permissions */}
        <div className="mb-8">
          <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-2">
            Access &amp; Permissions
          </div>
          <div className="border-t border-border">
            <SettingRow
              label="Models"
              description="Default list of models that new teams can access."
              isEditing={isEditing}
              viewContent={renderTags(values.models, getModelDisplayName)}
              editContent={
                <ModelSelect
                  value={editedValues.models || []}
                  onChange={(v) => update("models", v)}
                  context="global"
                  style={{ width: "100%" }}
                  options={{ includeSpecialOptions: true }}
                />
              }
            />

            <SettingRow
              label="Team Member Permissions"
              description="Default permissions granted to members of newly created teams. /key/info and /key/health are always included."
              isEditing={isEditing}
              viewContent={renderTags(values.team_member_permissions)}
              editContent={
                <div className="space-y-2">
                  <Select
                    value=""
                    onValueChange={(v) => {
                      if (v)
                        update("team_member_permissions", [
                          ...(editedValues.team_member_permissions ?? []),
                          v,
                        ]);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Add permission" />
                    </SelectTrigger>
                    <SelectContent>
                      {remainingPermissions.length === 0 ? (
                        <div className="py-2 px-3 text-sm text-muted-foreground">
                          No more permissions available
                        </div>
                      ) : (
                        remainingPermissions.map((option) => (
                          <SelectItem key={option} value={option}>
                            {option}
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                  {(editedValues.team_member_permissions ?? []).length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {(editedValues.team_member_permissions ?? []).map(
                        (p) => (
                          <Badge
                            key={p}
                            variant="default"
                            className="gap-1"
                          >
                            {p}
                            <button
                              type="button"
                              onClick={() =>
                                update(
                                  "team_member_permissions",
                                  (
                                    editedValues.team_member_permissions ?? []
                                  ).filter((x) => x !== p),
                                )
                              }
                              aria-label={`Remove ${p}`}
                            >
                              <X size={10} />
                            </button>
                          </Badge>
                        ),
                      )}
                    </div>
                  )}
                </div>
              }
            />
          </div>
        </div>
      </div>
    </Card>
  );
};

export default TeamSSOSettings;
