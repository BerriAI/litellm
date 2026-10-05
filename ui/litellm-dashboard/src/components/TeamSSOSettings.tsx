import { Edit, Save } from "lucide-react";
import React, { useEffect, useState } from "react";

import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Input } from "@/components/ui/input";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

import { getDefaultTeamSettings, updateDefaultTeamSettings, Organization } from "./networking";
import BudgetDurationDropdown, { getBudgetDurationLabel } from "./common_components/budget_duration_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { toast } from "@/lib/toast";
import { ModelSelect } from "./ModelSelect/ModelSelect";
import OrganizationDropdown from "./common_components/OrganizationDropdown";

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

const SettingRow: React.FC<SettingRowProps> = ({ label, description, isEditing, viewContent, editContent }) => (
  <div className="grid grid-cols-1 gap-3 border-b border-border py-5 last:border-b-0 md:grid-cols-3">
    <div className="pr-6">
      <p className="text-sm font-semibold text-foreground">{label}</p>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{description}</p>
    </div>
    <div className="flex items-center md:col-span-2">
      <div className="w-full">{isEditing ? editContent : viewContent}</div>
    </div>
  </div>
);

const NotSet = () => <span className="italic text-muted-foreground">Not set</span>;

const renderTags = (values: string[], displayFn?: (v: string) => string) => {
  if (!values || values.length === 0) return <NotSet />;
  return (
    <div className="flex flex-wrap gap-2">
      {values.map((v) => (
        <Badge key={v} variant="secondary">
          {displayFn ? displayFn(v) : v}
        </Badge>
      ))}
    </div>
  );
};

const getOrganizationLabel = (organizationId: string, organizations: Organization[] | undefined): string => {
  const organization = organizations?.find((org) => org.organization_id === organizationId);
  return organization?.organization_alias ? `${organization.organization_alias} (${organizationId})` : organizationId;
};

interface SettingsValues {
  max_budget: number | null;
  budget_duration: string | null;
  tpm_limit: number | null;
  rpm_limit: number | null;
  models: string[];
  team_member_permissions: string[];
  organization_id: string | null;
}

const DEFAULT_VALUES: SettingsValues = {
  max_budget: null,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  models: [],
  team_member_permissions: [],
  organization_id: null,
};

const TeamSSOSettings: React.FC<TeamSSOSettingsProps> = ({ accessToken }) => {
  const anchor = useComboboxAnchor();
  const [loading, setLoading] = useState<boolean>(true);
  const [values, setValues] = useState<SettingsValues>(DEFAULT_VALUES);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editedValues, setEditedValues] = useState<SettingsValues>(DEFAULT_VALUES);
  const [saving, setSaving] = useState<boolean>(false);
  const [fetchError, setFetchError] = useState<boolean>(false);
  const { data: organizations, isLoading: isOrganizationsLoading } = useOrganizations();

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
        toast.fromError("Failed to fetch team settings");
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
      const updatedSettings = await updateDefaultTeamSettings(accessToken, editedValues);
      const newValues = { ...DEFAULT_VALUES, ...(updatedSettings.settings || {}) };
      setValues(newValues);
      setEditedValues(newValues);
      setIsEditing(false);
      toast.success("Default team settings updated successfully");
    } catch (error) {
      console.error("Error updating team settings:", error);
      toast.fromError("Failed to update team settings");
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditedValues(values);
  };

  const update = <K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) => {
    setEditedValues((prev) => ({ ...prev, [key]: value }));
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center" aria-busy="true">
        <UiLoadingSpinner aria-label="Loading default team settings" />
      </div>
    );
  }

  if (fetchError) {
    return (
      <Card>
        <CardContent>
          <p>No team settings available or you do not have permission to view them.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="gap-0">
      <CardHeader className="gap-4 border-b border-border pb-6">
        <div>
          <CardTitle>
            <h3 className="text-lg font-semibold text-foreground">Default Team Settings</h3>
          </CardTitle>
          <CardDescription className="mt-1">
            These settings will be applied by default when creating new teams.
          </CardDescription>
        </div>
        <CardAction>
          {isEditing ? (
            <div className="flex gap-3">
              <Button type="button" variant="outline" onClick={handleCancel} disabled={saving}>
                Cancel
              </Button>
              <Button type="button" onClick={handleSave} disabled={saving}>
                {saving ? (
                  <UiLoadingSpinner className="size-4" aria-hidden="true" />
                ) : (
                  <Save data-icon="inline-start" />
                )}
                Save Changes
              </Button>
            </div>
          ) : (
            <Button type="button" variant="outline" onClick={() => setIsEditing(true)}>
              <Edit data-icon="inline-start" />
              Edit Settings
            </Button>
          )}
        </CardAction>
      </CardHeader>

      <CardContent className="pt-8">
        <section className="mb-8">
          <h4 className="mb-2 text-xs font-bold tracking-wider text-muted-foreground uppercase">
            Budget & Rate Limits
          </h4>
          <div className="border-t border-border">
            <SettingRow
              label="Max Budget"
              description="Maximum budget (in USD) for new automatically created teams."
              isEditing={isEditing}
              viewContent={
                values.max_budget != null ? <span>${Number(values.max_budget).toLocaleString()}</span> : <NotSet />
              }
              editContent={
                <InputGroup className="max-w-80">
                  <InputGroupAddon>$</InputGroupAddon>
                  <InputGroupInput
                    type="number"
                    step="any"
                    min={0}
                    value={editedValues.max_budget ?? ""}
                    onChange={(event) =>
                      update("max_budget", event.target.value === "" ? null : Number(event.target.value))
                    }
                    placeholder="Not set"
                    aria-label="Max Budget"
                  />
                </InputGroup>
              }
            />

            <SettingRow
              label="Budget Duration"
              description="How frequently the team's budget resets."
              isEditing={isEditing}
              viewContent={
                values.budget_duration ? <span>{getBudgetDurationLabel(values.budget_duration)}</span> : <NotSet />
              }
              editContent={
                <BudgetDurationDropdown
                  value={editedValues.budget_duration || null}
                  onChange={(v) => update("budget_duration", v ?? null)}
                  className="max-w-80"
                />
              }
            />

            <SettingRow
              label="TPM Limit"
              description="Maximum tokens per minute allowed across all models."
              isEditing={isEditing}
              viewContent={values.tpm_limit != null ? <span>{values.tpm_limit.toLocaleString()}</span> : <NotSet />}
              editContent={
                <Input
                  className="max-w-80"
                  type="number"
                  step={1}
                  value={editedValues.tpm_limit ?? ""}
                  onChange={(event) =>
                    update("tpm_limit", event.target.value === "" ? null : Number(event.target.value))
                  }
                  placeholder="Not set"
                  min={0}
                  aria-label="TPM Limit"
                />
              }
            />

            <SettingRow
              label="RPM Limit"
              description="Maximum requests per minute allowed across all models."
              isEditing={isEditing}
              viewContent={values.rpm_limit != null ? <span>{values.rpm_limit.toLocaleString()}</span> : <NotSet />}
              editContent={
                <Input
                  className="max-w-80"
                  type="number"
                  step={1}
                  value={editedValues.rpm_limit ?? ""}
                  onChange={(event) =>
                    update("rpm_limit", event.target.value === "" ? null : Number(event.target.value))
                  }
                  placeholder="Not set"
                  min={0}
                  aria-label="RPM Limit"
                />
              }
            />
          </div>
        </section>

        <section>
          <h4 className="mb-2 text-xs font-bold tracking-wider text-muted-foreground uppercase">
            Access & Permissions
          </h4>
          <div className="border-t border-border">
            <SettingRow
              label="Default Organization"
              description="Teams created without an explicit organization are assigned to this organization."
              isEditing={isEditing}
              viewContent={
                values.organization_id ? (
                  <span>{getOrganizationLabel(values.organization_id, organizations)}</span>
                ) : (
                  <NotSet />
                )
              }
              editContent={
                <div className="max-w-80 *:w-full">
                  <OrganizationDropdown
                    organizations={organizations}
                    loading={isOrganizationsLoading}
                    value={editedValues.organization_id ?? undefined}
                    onChange={(organizationId) => update("organization_id", organizationId || null)}
                    placeholder="Select an organization"
                  />
                </div>
              }
            />

            <SettingRow
              label="Models"
              description="Default list of models that new teams can access."
              isEditing={isEditing}
              viewContent={renderTags(values.models, getModelDisplayName)}
              editContent={
                <div className="*:w-full">
                  <ModelSelect
                    value={editedValues.models || []}
                    onChange={(v) => update("models", v)}
                    context="global"
                    options={{ includeSpecialOptions: true }}
                  />
                </div>
              }
            />

            <SettingRow
              label="Team Member Permissions"
              description="Default permissions granted to members of newly created teams. /key/info and /key/health are always included."
              isEditing={isEditing}
              viewContent={renderTags(values.team_member_permissions)}
              editContent={
                <Combobox
                  multiple
                  items={PERMISSION_OPTIONS}
                  value={editedValues.team_member_permissions || []}
                  onValueChange={(permissions: string[]) => update("team_member_permissions", permissions)}
                >
                  <ComboboxChips render={<div ref={anchor} />}>
                    <ComboboxValue>
                      {(permissions: string[]) =>
                        permissions.map((permission) => (
                          <ComboboxChip key={permission} aria-label={permission}>
                            {permission}
                          </ComboboxChip>
                        ))
                      }
                    </ComboboxValue>
                    <ComboboxChipsInput placeholder="Select permissions" aria-label="Team Member Permissions" />
                  </ComboboxChips>
                  <ComboboxContent anchor={anchor}>
                    <ComboboxList>
                      {(permission: string) => (
                        <ComboboxItem key={permission} value={permission}>
                          {permission}
                        </ComboboxItem>
                      )}
                    </ComboboxList>
                  </ComboboxContent>
                </Combobox>
              }
            />
          </div>
        </section>
      </CardContent>
    </Card>
  );
};

export default TeamSSOSettings;
