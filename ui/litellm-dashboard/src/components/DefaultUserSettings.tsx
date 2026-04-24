import React, { useEffect, useMemo, useState } from "react";
import { Plus as PlusOutlined, Trash2 as DeleteOutlined, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import {
  getInternalUserSettings,
  updateInternalUserSettings,
  modelAvailableCall,
} from "./networking";
import BudgetDurationDropdown, {
  getBudgetDurationLabel,
} from "./common_components/budget_duration_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import NotificationManager from "./molecules/notifications_manager";
import NumericalInput from "./shared/numerical_input";

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

/**
 * shadcn multi-select (Select + badge chips) used for enum arrays.
 */
function MultiSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No more options
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

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

        if (accessToken) {
          try {
            const modelResponse = await modelAvailableCall(
              accessToken,
              userID,
              userRole,
            );
            if (modelResponse && modelResponse.data) {
              const modelNames = modelResponse.data.map(
                (model: { id: string }) => model.id,
              );
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const handleSaveSettings = async () => {
    if (!accessToken) return;

    setSaving(true);
    try {
      const processedValues = Object.entries(editedValues).reduce(
        (acc, [key, value]) => {
          acc[key] = value === "" ? null : value;
          return acc;
        },
        {} as Record<string, any>,
      );

      const updatedSettings = await updateInternalUserSettings(
        accessToken,
        processedValues,
      );
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

  const renderTeamsEditor = (teams: any[]) => {
    const normalizedTeams = normalizeTeams(teams);

    const updateTeam = (index: number, field: keyof TeamEntry, value: any) => {
      const updatedTeams = [...normalizedTeams];
      updatedTeams[index] = {
        ...updatedTeams[index],
        [field]: value,
      };
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
          <div
            key={index}
            className="border border-border rounded-lg p-4 bg-muted/50"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="font-medium">Team {index + 1}</span>
              <Button
                size="sm"
                variant="destructive"
                onClick={() => removeTeam(index)}
              >
                <DeleteOutlined className="h-4 w-4 mr-1" />
                Remove
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <span className="text-sm font-medium mb-1 block">Team ID</span>
                <Input
                  value={team.team_id}
                  onChange={(e) =>
                    updateTeam(index, "team_id", e.target.value)
                  }
                  placeholder="Enter team ID"
                />
              </div>

              <div>
                <span className="text-sm font-medium mb-1 block">
                  Max Budget in Team
                </span>
                <NumericalInput
                  value={team.max_budget_in_team}
                  onChange={(value: number | null) =>
                    updateTeam(index, "max_budget_in_team", value)
                  }
                  placeholder="Optional"
                  min={0}
                  step={0.01}
                  precision={2}
                />
              </div>

              <div>
                <span className="text-sm font-medium mb-1 block">
                  User Role
                </span>
                <Select
                  value={team.user_role}
                  onValueChange={(value) =>
                    updateTeam(index, "user_role", value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="user">User</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        ))}

        <Button variant="outline" onClick={addTeam} className="w-full">
          <PlusOutlined className="h-4 w-4 mr-1" />
          Add Team
        </Button>
      </div>
    );
  };

  const renderEditableField = (key: string, property: any, value: any) => {
    const type = property.type;

    if (key === "teams") {
      return (
        <div className="mt-2">{renderTeamsEditor(editedValues[key] || [])}</div>
      );
    } else if (key === "user_role" && possibleUIRoles) {
      return (
        <Select
          value={editedValues[key] || ""}
          onValueChange={(v) => handleTextInputChange(key, v)}
        >
          <SelectTrigger className="mt-2">
            <SelectValue placeholder="Select a role" />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(possibleUIRoles)
              .filter(([role]) => role.includes("internal_user"))
              .map(([role, { ui_label, description }]) => (
                <SelectItem key={role} value={role}>
                  <div className="flex items-center">
                    <span>{ui_label}</span>
                    <span className="ml-2 text-xs text-muted-foreground">
                      {description}
                    </span>
                  </div>
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
            onCheckedChange={(checked) => handleTextInputChange(key, checked)}
          />
        </div>
      );
    } else if (type === "array" && property.items?.enum) {
      const options = property.items.enum.map((option: string) => ({
        label: option,
        value: option,
      }));
      return (
        <div className="mt-2">
          <MultiSelect
            value={editedValues[key] || []}
            onChange={(v) => handleTextInputChange(key, v)}
            options={options}
            placeholder="Select options"
          />
        </div>
      );
    } else if (key === "models") {
      const options = [
        { label: "No Default Models", value: "no-default-models" },
        { label: "All Proxy Models", value: "all-proxy-models" },
        ...availableModels.map((model) => ({
          label: getModelDisplayName(model),
          value: model,
        })),
      ];
      return (
        <div className="mt-2">
          <MultiSelect
            value={editedValues[key] || []}
            onChange={(v) => handleTextInputChange(key, v)}
            options={options}
            placeholder="Select models"
          />
        </div>
      );
    } else if (type === "string" && property.enum) {
      return (
        <Select
          value={editedValues[key] || ""}
          onValueChange={(v) => handleTextInputChange(key, v)}
        >
          <SelectTrigger className="mt-2">
            <SelectValue placeholder="Select an option" />
          </SelectTrigger>
          <SelectContent>
            {property.enum.map((option: string) => (
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
          value={
            editedValues[key] !== undefined ? String(editedValues[key]) : ""
          }
          onChange={(e) => handleTextInputChange(key, e.target.value)}
          placeholder={property.description || ""}
          className="mt-2"
        />
      );
    }
  };

  const renderValue = (key: string, value: any): JSX.Element => {
    if (value === null || value === undefined)
      return <span className="text-muted-foreground">Not set</span>;

    if (key === "teams" && Array.isArray(value)) {
      if (value.length === 0)
        return (
          <span className="text-muted-foreground">No teams assigned</span>
        );

      const normalizedTeams = normalizeTeams(value);

      return (
        <div className="space-y-2 mt-1">
          {normalizedTeams.map((team, index) => (
            <div
              key={index}
              className="border border-border rounded-lg p-3 bg-background"
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
                <div>
                  <span className="font-medium text-muted-foreground">
                    Team ID:
                  </span>
                  <p className="text-foreground">
                    {team.team_id || "Not specified"}
                  </p>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">
                    Max Budget:
                  </span>
                  <p className="text-foreground">
                    {team.max_budget_in_team !== undefined
                      ? `$${formatNumberWithCommas(team.max_budget_in_team, 4)}`
                      : "No limit"}
                  </p>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">
                    Role:
                  </span>
                  <p className="text-foreground capitalize">{team.user_role}</p>
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
          {description && (
            <p className="text-xs text-muted-foreground mt-1">{description}</p>
          )}
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
      if (value.length === 0)
        return <span className="text-muted-foreground">None</span>;

      return (
        <div className="flex flex-wrap gap-2 mt-1">
          {value.map((model, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-blue-100 dark:bg-blue-950/50 text-blue-800 dark:text-blue-300 rounded text-xs"
            >
              {getModelDisplayName(model)}
            </span>
          ))}
        </div>
      );
    }

    if (typeof value === "object") {
      if (Array.isArray(value)) {
        if (value.length === 0)
          return <span className="text-muted-foreground">None</span>;

        return (
          <div className="flex flex-wrap gap-2 mt-1">
            {value.map((item, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-blue-100 dark:bg-blue-950/50 text-blue-800 dark:text-blue-300 rounded text-xs"
              >
                {typeof item === "object" ? JSON.stringify(item) : String(item)}
              </span>
            ))}
          </div>
        );
      }

      return (
        <pre className="bg-muted p-2 rounded text-xs overflow-auto mt-1">
          {JSON.stringify(value, null, 2)}
        </pre>
      );
    }

    return <span>{String(value)}</span>;
  };

  if (loading) {
    return (
      <div className="flex flex-col gap-3 p-6">
        <Skeleton className="h-6 w-48" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }

  if (!settings) {
    return (
      <Card className="p-6">
        <span>
          No settings available or you do not have permission to view them.
        </span>
      </Card>
    );
  }

  const renderSettings = () => {
    const { values, field_schema } = settings;

    if (!field_schema || !field_schema.properties) {
      return <span>No schema information available</span>;
    }

    return Object.entries(field_schema.properties).map(
      ([key, property]: [string, any]) => {
        const value = values[key];
        const displayName = key
          .replace(/_/g, " ")
          .replace(/\b\w/g, (l) => l.toUpperCase());

        return (
          <div
            key={key}
            className="mb-6 pb-6 border-b border-border last:border-0"
          >
            <span className="font-medium text-lg block">{displayName}</span>
            <p className="text-sm text-muted-foreground mt-1">
              {property.description || "No description available"}
            </p>

            {isEditing ? (
              <div className="mt-2">
                {renderEditableField(key, property, value)}
              </div>
            ) : (
              <div className="mt-1 p-2 bg-muted/50 rounded">
                {renderValue(key, value)}
              </div>
            )}
          </div>
        );
      },
    );
  };

  return (
    <Card className="p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold m-0">Default User Settings</h2>
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
                {saving ? "Saving..." : "Save Changes"}
              </Button>
            </div>
          ) : (
            <Button onClick={() => setIsEditing(true)}>Edit Settings</Button>
          ))}
      </div>

      {settings?.field_schema?.description && (
        <p className="mb-4">{settings.field_schema.description}</p>
      )}
      <Separator />

      <div className="mt-4 space-y-4">{renderSettings()}</div>
    </Card>
  );
};

export default DefaultUserSettings;
