import React, { useState, useEffect } from "react";
import { Plus, Trash2 } from "lucide-react";
import { getInternalUserSettings, updateInternalUserSettings, modelAvailableCall } from "@/components/networking";
import BudgetDurationDropdown, {
  getBudgetDurationLabel,
} from "@/components/common_components/budget_duration_dropdown";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import NotificationManager from "@/components/molecules/notifications_manager";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

interface DefaultUserSettingsProps {
  accessToken: string | null;
  possibleUIRoles?: Record<string, Record<string, string>> | null;
  userID: string;
  userRole: string;
}

interface TeamEntry {
  team_id: string;
  max_budget_in_team?: number;
  user_role: "user" | "admin";
}

const TEAM_MEMBER_ROLES: TeamEntry["user_role"][] = ["user", "admin"];

const TEAM_MEMBER_ROLE_LABELS: Record<TeamEntry["user_role"], string> = {
  user: "User",
  admin: "Admin",
};

const DefaultUserSettings: React.FC<DefaultUserSettingsProps> = ({
  accessToken,
  possibleUIRoles,
  userID,
  userRole,
}) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [settings, setSettings] = useState<any>(null);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editedValues, setEditedValues] = useState<any>({});
  const [saving, setSaving] = useState<boolean>(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  useEffect(() => {
    const fetchSSOSettings = async () => {
      if (!accessToken) {
        setLoading(false);
        return;
      }

      try {
        const data = await getInternalUserSettings(accessToken);
        setSettings(data);
        setEditedValues(data.values || {});

        // Fetch available models
        if (accessToken) {
          try {
            const modelResponse = await modelAvailableCall(accessToken, userID, userRole);
            if (modelResponse && modelResponse.data) {
              const modelNames = modelResponse.data.map((model: { id: string }) => model.id);
              setAvailableModels(modelNames);
            }
          } catch (error) {
            console.error("Error fetching available models:", error);
          }
        }
      } catch (error) {
        console.error("Error fetching SSO settings:", error);
        NotificationManager.fromBackend("Failed to fetch SSO settings");
      } finally {
        setLoading(false);
      }
    };

    fetchSSOSettings();
  }, [accessToken]);

  const handleSaveSettings = async () => {
    if (!accessToken) return;

    setSaving(true);
    try {
      // Convert empty strings to null
      const processedValues = Object.entries(editedValues).reduce(
        (acc, [key, value]) => {
          acc[key] = value === "" ? null : value;
          return acc;
        },
        {} as Record<string, any>,
      );

      const updatedSettings = await updateInternalUserSettings(accessToken, processedValues);
      setSettings({ ...settings, values: updatedSettings.settings });
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating SSO settings:", error);
      NotificationManager.fromBackend("Failed to update settings: " + error);
    } finally {
      setSaving(false);
    }
  };

  const handleTextInputChange = (key: string, value: any) => {
    setEditedValues((prev: Record<string, any>) => ({
      ...prev,
      [key]: value,
    }));
  };

  // Helper function to normalize teams array to consistent format
  const normalizeTeams = (teams: any[]): TeamEntry[] => {
    if (!teams || !Array.isArray(teams)) return [];

    return teams.map((team) => {
      if (typeof team === "string") {
        return {
          team_id: team,
          user_role: "user" as const,
        };
      } else if (typeof team === "object" && team.team_id) {
        return {
          team_id: team.team_id,
          max_budget_in_team: team.max_budget_in_team,
          user_role: team.user_role || "user",
        };
      }
      return {
        team_id: "",
        user_role: "user" as const,
      };
    });
  };

  const renderMultiSelect = (key: string, options: string[], placeholder: string, getLabel: (o: string) => string) => {
    const selected: string[] = editedValues[key] || [];

    return (
      <Combobox
        multiple
        items={options}
        value={selected}
        onValueChange={(value: string[]) => handleTextInputChange(key, value)}
      >
        <ComboboxChips className="mt-2">
          <ComboboxValue>
            {(values: string[]) =>
              values.map((option) => (
                <ComboboxChip key={option} aria-label={getLabel(option)}>
                  {getLabel(option)}
                </ComboboxChip>
              ))
            }
          </ComboboxValue>
          <ComboboxChipsInput placeholder={placeholder} className="border-0 bg-transparent" />
        </ComboboxChips>
        <ComboboxContent>
          <ComboboxEmpty>No options found</ComboboxEmpty>
          <ComboboxList>
            {(option: string) => (
              <ComboboxItem key={option} value={option}>
                {getLabel(option)}
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
    );
  };

  // Teams editor component
  const renderTeamsEditor = (teams: any[]) => {
    const normalizedTeams = normalizeTeams(teams);

    const updateTeam = (index: number, field: keyof TeamEntry, value: any) => {
      const updatedTeams = normalizedTeams.map((team, i) => (i === index ? { ...team, [field]: value } : team));
      handleTextInputChange("teams", updatedTeams);
    };

    const addTeam = () => {
      const newTeam: TeamEntry = {
        team_id: "",
        user_role: "user",
      };
      handleTextInputChange("teams", [...normalizedTeams, newTeam]);
    };

    const removeTeam = (index: number) => {
      const updatedTeams = normalizedTeams.filter((_, i) => i !== index);
      handleTextInputChange("teams", updatedTeams);
    };

    return (
      <div className="space-y-3">
        {normalizedTeams.map((team, index) => (
          <div key={index} className="rounded-lg border border-border bg-muted/40 p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="font-medium">Team {index + 1}</p>
              <Button size="sm" variant="destructive" onClick={() => removeTeam(index)}>
                <Trash2 />
                Remove
              </Button>
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div>
                <p className="mb-1 text-sm font-medium">Team ID</p>
                <Input
                  value={team.team_id}
                  onChange={(event) => updateTeam(index, "team_id", event.target.value)}
                  placeholder="Enter team ID"
                />
              </div>

              <div>
                <p className="mb-1 text-sm font-medium">Max Budget in Team</p>
                <Input
                  type="number"
                  value={team.max_budget_in_team ?? ""}
                  onChange={(event) =>
                    updateTeam(
                      index,
                      "max_budget_in_team",
                      event.target.value === "" ? undefined : Number(event.target.value),
                    )
                  }
                  placeholder="Optional"
                  min={0}
                  step={0.01}
                />
              </div>

              <div>
                <p className="mb-1 text-sm font-medium">User Role</p>
                <Select
                  value={team.user_role}
                  onValueChange={(value: TeamEntry["user_role"] | null) =>
                    updateTeam(index, "user_role", value ?? "user")
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue>{TEAM_MEMBER_ROLE_LABELS[team.user_role]}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {TEAM_MEMBER_ROLES.map((role) => (
                      <SelectItem key={role} value={role}>
                        {TEAM_MEMBER_ROLE_LABELS[role]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        ))}

        <Button variant="outline" onClick={addTeam} className="w-full">
          <Plus />
          Add Team
        </Button>
      </div>
    );
  };

  const renderEditableField = (key: string, property: any, value: any) => {
    const type = property.type;

    if (key === "teams") {
      return <div className="mt-2">{renderTeamsEditor(editedValues[key] || [])}</div>;
    } else if (key === "user_role" && possibleUIRoles) {
      const internalUserRoles = Object.entries(possibleUIRoles).filter(([role]) => role.includes("internal_user"));
      const selectedRole = editedValues[key] || null;

      return (
        <Select value={selectedRole} onValueChange={(role: string) => handleTextInputChange(key, role)}>
          <SelectTrigger className="mt-2 w-full">
            <SelectValue>
              {selectedRole ? possibleUIRoles[selectedRole]?.ui_label || selectedRole : "Select a role"}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {internalUserRoles.map(([role, { ui_label, description }]) => (
              <SelectItem key={role} value={role}>
                <span>{ui_label}</span>
                <span className="ml-2 text-xs text-muted-foreground">{description}</span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    } else if (key === "budget_duration") {
      return (
        <BudgetDurationDropdown
          value={editedValues[key] || null}
          onChange={(value) => handleTextInputChange(key, value)}
          className="mt-2"
        />
      );
    } else if (type === "boolean") {
      return (
        <div className="mt-2">
          <Switch
            checked={!!editedValues[key]}
            onCheckedChange={(checked: boolean) => handleTextInputChange(key, checked)}
          />
        </div>
      );
    } else if (type === "array" && property.items?.enum) {
      return renderMultiSelect(key, property.items.enum as string[], "Select options", (option) => option);
    } else if (key === "models") {
      return renderMultiSelect(
        key,
        ["no-default-models", "all-proxy-models", ...availableModels],
        "Select models",
        (model) => {
          if (model === "no-default-models") return "No Default Models";
          if (model === "all-proxy-models") return "All Proxy Models";
          return getModelDisplayName(model);
        },
      );
    } else if (type === "string" && property.enum) {
      const selected = editedValues[key] || null;

      return (
        <Select value={selected} onValueChange={(option: string) => handleTextInputChange(key, option)}>
          <SelectTrigger className="mt-2 w-full">
            <SelectValue placeholder={`Select ${key.replace(/_/g, " ")}`} />
          </SelectTrigger>
          <SelectContent>
            {(property.enum as string[]).map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    } else {
      return (
        <Input
          value={editedValues[key] !== undefined && editedValues[key] !== null ? String(editedValues[key]) : ""}
          onChange={(event) => handleTextInputChange(key, event.target.value)}
          placeholder={property.description || ""}
          className="mt-2"
        />
      );
    }
  };

  const renderValue = (key: string, value: any): JSX.Element => {
    if (value === null || value === undefined) return <span className="text-muted-foreground">Not set</span>;

    if (key === "teams" && Array.isArray(value)) {
      if (value.length === 0) return <span className="text-muted-foreground">No teams assigned</span>;

      const normalizedTeams = normalizeTeams(value);

      return (
        <div className="mt-1 space-y-2">
          {normalizedTeams.map((team, index) => (
            <div key={index} className="rounded-lg border border-border bg-card p-3">
              <div className="grid grid-cols-1 gap-2 text-sm md:grid-cols-3">
                <div>
                  <span className="font-medium text-muted-foreground">Team ID:</span>
                  <p>{team.team_id || "Not specified"}</p>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">Max Budget:</span>
                  <p>
                    {team.max_budget_in_team !== undefined
                      ? `$${formatNumberWithCommas(team.max_budget_in_team, 4)}`
                      : "No limit"}
                  </p>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">Role:</span>
                  <p className="capitalize">{team.user_role}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      );
    }

    if (key === "user_role" && possibleUIRoles && possibleUIRoles[value]) {
      const { ui_label, description } = possibleUIRoles[value];
      return (
        <div>
          <span className="font-medium">{ui_label}</span>
          {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
        </div>
      );
    }

    if (key === "budget_duration") {
      return <span>{getBudgetDurationLabel(value)}</span>;
    }

    if (typeof value === "boolean") {
      return <span>{value ? "Enabled" : "Disabled"}</span>;
    }

    if (key === "models" && Array.isArray(value)) {
      if (value.length === 0) return <span className="text-muted-foreground">None</span>;

      return (
        <div className="mt-1 flex flex-wrap gap-2">
          {value.map((model, index) => (
            <Badge key={index} variant="secondary">
              {getModelDisplayName(model)}
            </Badge>
          ))}
        </div>
      );
    }

    if (typeof value === "object") {
      if (Array.isArray(value)) {
        if (value.length === 0) return <span className="text-muted-foreground">None</span>;

        return (
          <div className="mt-1 flex flex-wrap gap-2">
            {value.map((item, index) => (
              <Badge key={index} variant="secondary">
                {typeof item === "object" ? JSON.stringify(item) : String(item)}
              </Badge>
            ))}
          </div>
        );
      }

      return <pre className="mt-1 overflow-auto rounded-sm bg-muted p-2 text-xs">{JSON.stringify(value, null, 2)}</pre>;
    }

    return <span>{String(value)}</span>;
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <UiLoadingSpinner />
      </div>
    );
  }

  if (!settings) {
    return (
      <Card>
        <CardContent>
          <p>No settings available or you do not have permission to view them.</p>
        </CardContent>
      </Card>
    );
  }

  // Dynamically render settings based on the schema
  const renderSettings = () => {
    const { values, field_schema } = settings;

    if (!field_schema || !field_schema.properties) {
      return <p>No schema information available</p>;
    }

    return Object.entries(field_schema.properties).map(([key, property]: [string, any]) => {
      const value = values[key];
      const displayName = key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());

      return (
        <div key={key} className="mb-6 border-b border-border pb-6 last:border-0">
          <p className="text-lg font-medium">{displayName}</p>
          <p className="mt-1 text-sm text-muted-foreground">{property.description || "No description available"}</p>

          {isEditing ? (
            <div className="mt-2">{renderEditableField(key, property, value)}</div>
          ) : (
            <div className="mt-1 rounded-sm bg-muted p-2">{renderValue(key, value)}</div>
          )}
        </div>
      );
    });
  };

  return (
    <Card>
      <CardContent>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Default User Settings</h2>
          {!loading &&
            settings &&
            (isEditing ? (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => {
                    setIsEditing(false);
                    setEditedValues(settings.values || {});
                  }}
                  disabled={saving}
                >
                  Cancel
                </Button>
                <Button onClick={handleSaveSettings} disabled={saving}>
                  {saving && <UiLoadingSpinner className="size-4" />}
                  Save Changes
                </Button>
              </div>
            ) : (
              <Button onClick={() => setIsEditing(true)}>Edit Settings</Button>
            ))}
        </div>

        {settings?.field_schema?.description && <p className="mb-4">{settings.field_schema.description}</p>}
        <Separator />

        <div className="mt-4 space-y-4">{renderSettings()}</div>
      </CardContent>
    </Card>
  );
};

export default DefaultUserSettings;
