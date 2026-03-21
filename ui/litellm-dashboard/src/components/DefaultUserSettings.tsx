import React, { useState, useEffect } from "react";
import { Card, Title, Text, Divider, TextInput } from "@tremor/react";
import { Button, Typography, Spin, Switch, Select, InputNumber } from "antd";
import { PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import { getInternalUserSettings, updateInternalUserSettings, modelAvailableCall } from "./networking";
import BudgetDurationDropdown, { getBudgetDurationLabel } from "./common_components/budget_duration_dropdown";
import ConfirmSettingsChangeModal, { SettingsChange } from "./common_components/ConfirmSettingsChangeModal";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import NotificationManager from "./molecules/notifications_manager";

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

/** Shape returned by the GET endpoint and sent back on PATCH. */
interface DefaultUserSettingsValues {
  user_role?: string | null;
  max_budget?: number | null;
  budget_duration?: string | null;
  models?: string[] | null;
  teams?: (string | TeamEntry)[] | null;
}

interface SettingsResponse {
  values: DefaultUserSettingsValues;
  field_schema: {
    description?: string;
    properties: Record<string, { type: string; description?: string; enum?: string[]; items?: { enum?: string[] } }>;
  };
}

/** Normalize the backend's union type (string | TeamEntry) to TeamEntry[]. */
function normalizeTeams(teams: (string | TeamEntry)[]): TeamEntry[] {
  return teams.map((team) =>
    typeof team === "string"
      ? { team_id: team, user_role: "user" as const }
      : { team_id: team.team_id, max_budget_in_team: team.max_budget_in_team, user_role: team.user_role || "user" },
  );
}

/**
 * Compare original settings values against edited values and return a list of
 * human-readable changes. A change is "destructive" if something was removed,
 * cleared, or reduced.
 */
export function computeSettingsDiff(
  original: DefaultUserSettingsValues,
  edited: DefaultUserSettingsValues,
): { changes: SettingsChange[]; hasDestructiveChanges: boolean } {
  const changes: SettingsChange[] = [];

  // --- Teams ---
  const oldTeams = normalizeTeams(original.teams || []);
  const newTeams = normalizeTeams(edited.teams || []);
  const oldTeamMap = new Map(oldTeams.map((t) => [t.team_id, t]));
  const newTeamIds = new Set(newTeams.map((t) => t.team_id));

  for (const t of oldTeams) {
    if (!newTeamIds.has(t.team_id)) {
      changes.push({ field: "Teams", type: "removed", details: `Team "${t.team_id}" removed` });
    }
  }
  for (const t of newTeams) {
    if (!t.team_id) continue;
    const old = oldTeamMap.get(t.team_id);
    if (!old) {
      changes.push({ field: "Teams", type: "added", details: `Team "${t.team_id}" added` });
      continue;
    }
    if (old.max_budget_in_team !== t.max_budget_in_team) {
      const fmt = (v?: number) => (v !== undefined ? `$${v}` : "No limit");
      changes.push({ field: "Teams", type: "changed", details: `Team "${t.team_id}" budget: ${fmt(old.max_budget_in_team)} → ${fmt(t.max_budget_in_team)}` });
    }
    if (old.user_role !== t.user_role) {
      changes.push({ field: "Teams", type: "changed", details: `Team "${t.team_id}" role: ${old.user_role} → ${t.user_role}` });
    }
  }

  // --- Models ---
  const oldModels = new Set(original.models || []);
  const newModels = new Set(edited.models || []);
  const removedModels = [...oldModels].filter((m) => !newModels.has(m));
  const addedModels = [...newModels].filter((m) => !oldModels.has(m));
  if (removedModels.length) changes.push({ field: "Models", type: "removed", details: `${removedModels.join(", ")} removed` });
  if (addedModels.length) changes.push({ field: "Models", type: "added", details: `${addedModels.join(", ")} added` });

  // --- Scalars ---
  const scalarKeys = ["user_role", "max_budget", "budget_duration"] as const;
  for (const key of scalarKeys) {
    const oldVal = original[key] ?? null;
    const newVal = edited[key] ?? null;
    if (oldVal === newVal) continue;

    const displayName = key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
    if (oldVal != null && newVal == null) {
      changes.push({ field: displayName, type: "removed", details: `Cleared (was "${oldVal}")` });
    } else if (oldVal == null && newVal != null) {
      changes.push({ field: displayName, type: "added", details: `Set to "${newVal}"` });
    } else {
      changes.push({ field: displayName, type: "changed", details: `"${oldVal}" → "${newVal}"` });
    }
  }

  return { changes, hasDestructiveChanges: changes.some((c) => c.type === "removed" || c.type === "changed") };
}

const DefaultUserSettings: React.FC<DefaultUserSettingsProps> = ({
  accessToken,
  possibleUIRoles,
  userID,
  userRole,
}) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editedValues, setEditedValues] = useState<DefaultUserSettingsValues>({});
  const [saving, setSaving] = useState<boolean>(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showConfirmModal, setShowConfirmModal] = useState<boolean>(false);
  const [pendingChanges, setPendingChanges] = useState<SettingsChange[]>([]);
  const [pendingProcessedValues, setPendingProcessedValues] = useState<DefaultUserSettingsValues | null>(null);
  const { Paragraph } = Typography;
  const { Option } = Select;

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

  /** Perform the actual API save with the given processed values. */
  const executeSave = async (processedValues: DefaultUserSettingsValues) => {
    if (!accessToken) return;

    setSaving(true);
    try {
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

  /** Called when user clicks "Save Changes". Shows modal if destructive. */
  const handleSaveSettings = async () => {
    if (!accessToken) return;

    // Convert empty strings to null
    const processedValues = Object.fromEntries(
      Object.entries(editedValues).map(([key, value]) => [key, value === "" ? null : value]),
    ) as DefaultUserSettingsValues;

    const { changes, hasDestructiveChanges } = computeSettingsDiff(
      settings?.values || {},
      processedValues,
    );

    if (hasDestructiveChanges) {
      setPendingChanges(changes);
      setPendingProcessedValues(processedValues);
      setShowConfirmModal(true);
      return;
    }

    // No destructive changes — save directly
    await executeSave(processedValues);
  };

  /** Called when user confirms changes in the modal. */
  const handleConfirmSave = async () => {
    if (!pendingProcessedValues) return;
    await executeSave(pendingProcessedValues);
    setShowConfirmModal(false);
    setPendingChanges([]);
    setPendingProcessedValues(null);
  };

  const handleTextInputChange = (key: keyof DefaultUserSettingsValues, value: DefaultUserSettingsValues[typeof key]) => {
    setEditedValues((prev) => ({ ...prev, [key]: value }));
  };

  // Teams editor component
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
          <div key={index} className="border rounded-lg p-4 bg-gray-50">
            <div className="flex items-center justify-between mb-3">
              <Text className="font-medium">Team {index + 1}</Text>
              <Button
                size="small"
                danger
                icon={<DeleteOutlined />}
                onClick={() => removeTeam(index)}
              >
                Remove
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <Text className="text-sm font-medium mb-1">Team ID</Text>
                <TextInput
                  value={team.team_id}
                  onChange={(e) => updateTeam(index, "team_id", e.target.value)}
                  placeholder="Enter team ID"
                />
              </div>

              <div>
                <Text className="text-sm font-medium mb-1">Max Budget in Team</Text>
                <InputNumber
                  style={{ width: "100%" }}
                  value={team.max_budget_in_team}
                  onChange={(value) => updateTeam(index, "max_budget_in_team", value)}
                  placeholder="Optional"
                  min={0}
                  step={0.01}
                  precision={2}
                />
              </div>

              <div>
                <Text className="text-sm font-medium mb-1">User Role</Text>
                <Select
                  style={{ width: "100%" }}
                  value={team.user_role}
                  onChange={(value) => updateTeam(index, "user_role", value)}
                >
                  <Option value="user">User</Option>
                  <Option value="admin">Admin</Option>
                </Select>
              </div>
            </div>
          </div>
        ))}

        <Button icon={<PlusOutlined />} onClick={addTeam} className="w-full">
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
      return (
        <Select
          style={{ width: "100%" }}
          value={editedValues[key] || ""}
          onChange={(value) => handleTextInputChange(key, value)}
          className="mt-2"
        >
          {Object.entries(possibleUIRoles)
            .filter(([role]) => role.includes("internal_user"))
            .map(([role, { ui_label, description }]) => (
              <Option key={role} value={role}>
                <div className="flex items-center">
                  <span>{ui_label}</span>
                  <span className="ml-2 text-xs text-gray-500">{description}</span>
                </div>
              </Option>
            ))}
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
          <Switch checked={!!editedValues[key]} onChange={(checked) => handleTextInputChange(key, checked)} />
        </div>
      );
    } else if (type === "array" && property.items?.enum) {
      return (
        <Select
          mode="multiple"
          style={{ width: "100%" }}
          value={editedValues[key] || []}
          onChange={(value) => handleTextInputChange(key, value)}
          className="mt-2"
        >
          {property.items.enum.map((option: string) => (
            <Option key={option} value={option}>
              {option}
            </Option>
          ))}
        </Select>
      );
    } else if (key === "models") {
      return (
        <Select
          mode="multiple"
          style={{ width: "100%" }}
          value={editedValues[key] || []}
          onChange={(value) => handleTextInputChange(key, value)}
          className="mt-2"
        >
          <Option key="no-default-models" value="no-default-models">
            No Default Models
          </Option>
          <Option key="all-proxy-models" value="all-proxy-models">
            All Proxy Models
          </Option>
          {availableModels.map((model: string) => (
            <Option key={model} value={model}>
              {getModelDisplayName(model)}
            </Option>
          ))}
        </Select>
      );
    } else if (type === "number") {
      return (
        <InputNumber
          style={{ width: "100%" }}
          value={editedValues[key] as number | undefined}
          onChange={(value) => handleTextInputChange(key, value)}
          placeholder={property.description || ""}
          className="mt-2"
          min={0}
          step={0.01}
          precision={2}
        />
      );
    } else if (type === "string" && property.enum) {
      return (
        <Select
          style={{ width: "100%" }}
          value={editedValues[key] || ""}
          onChange={(value) => handleTextInputChange(key, value)}
          className="mt-2"
        >
          {property.enum.map((option: string) => (
            <Option key={option} value={option}>
              {option}
            </Option>
          ))}
        </Select>
      );
    } else {
      return (
        <TextInput
          value={editedValues[key] !== undefined ? String(editedValues[key]) : ""}
          onChange={(e) => handleTextInputChange(key, e.target.value)}
          placeholder={property.description || ""}
          className="mt-2"
        />
      );
    }
  };

  const renderValue = (key: string, value: any): JSX.Element => {
    if (value === null || value === undefined) return <span className="text-gray-400">Not set</span>;

    if (key === "teams" && Array.isArray(value)) {
      if (value.length === 0) return <span className="text-gray-400">No teams assigned</span>;

      const normalizedTeams = normalizeTeams(value);

      return (
        <div className="space-y-2 mt-1">
          {normalizedTeams.map((team, index) => (
            <div key={index} className="border rounded-lg p-3 bg-white">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
                <div>
                  <span className="font-medium text-gray-600">Team ID:</span>
                  <p className="text-gray-900">{team.team_id || "Not specified"}</p>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Max Budget:</span>
                  <p className="text-gray-900">
                    {team.max_budget_in_team !== undefined
                      ? `$${formatNumberWithCommas(team.max_budget_in_team, 4)}`
                      : "No limit"}
                  </p>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Role:</span>
                  <p className="text-gray-900 capitalize">{team.user_role}</p>
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
          {description && <p className="text-xs text-gray-500 mt-1">{description}</p>}
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
      if (value.length === 0) return <span className="text-gray-400">None</span>;

      return (
        <div className="flex flex-wrap gap-2 mt-1">
          {value.map((model, index) => (
            <span key={index} className="px-2 py-1 bg-blue-100 rounded text-xs">
              {getModelDisplayName(model)}
            </span>
          ))}
        </div>
      );
    }

    if (typeof value === "object") {
      if (Array.isArray(value)) {
        if (value.length === 0) return <span className="text-gray-400">None</span>;

        return (
          <div className="flex flex-wrap gap-2 mt-1">
            {value.map((item, index) => (
              <span key={index} className="px-2 py-1 bg-blue-100 rounded text-xs">
                {typeof item === "object" ? JSON.stringify(item) : String(item)}
              </span>
            ))}
          </div>
        );
      }

      return <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">{JSON.stringify(value, null, 2)}</pre>;
    }

    return <span>{String(value)}</span>;
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Spin size="large" />
      </div>
    );
  }

  if (!settings) {
    return (
      <Card>
        <Text>No settings available or you do not have permission to view them.</Text>
      </Card>
    );
  }

  // Dynamically render settings based on the schema
  const renderSettings = () => {
    const { values, field_schema } = settings;

    if (!field_schema || !field_schema.properties) {
      return <Text>No schema information available</Text>;
    }

    return Object.entries(field_schema.properties).map(([key, property]: [string, any]) => {
      const value = values[key];
      const displayName = key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());

      return (
        <div key={key} className="mb-6 pb-6 border-b border-gray-200 last:border-0">
          <Text className="font-medium text-lg">{displayName}</Text>
          <Paragraph className="text-sm text-gray-500 mt-1">
            {property.description || "No description available"}
          </Paragraph>

          {isEditing ? (
            <div className="mt-2">{renderEditableField(key, property, value)}</div>
          ) : (
            <div className="mt-1 p-2 bg-gray-50 rounded">{renderValue(key, value)}</div>
          )}
        </div>
      );
    });
  };

  return (
    <Card>
      <div className="flex justify-between items-center mb-4">
        <Title>Default User Settings</Title>
        {!loading &&
          settings &&
          (isEditing ? (
            <div className="flex gap-2">
              <Button
                onClick={() => {
                  setIsEditing(false);
                  setEditedValues(settings.values || {});
                }}
                disabled={saving}
              >
                Cancel
              </Button>
              <Button type="primary" onClick={handleSaveSettings} loading={saving}>
                Save Changes
              </Button>
            </div>
          ) : (
            <Button type="primary" onClick={() => setIsEditing(true)}>Edit Settings</Button>
          ))}
      </div>

      {settings?.field_schema?.description && (
        <Paragraph className="mb-4">{settings.field_schema.description}</Paragraph>
      )}
      <Divider />

      <div className="mt-4 space-y-4">{renderSettings()}</div>

      <ConfirmSettingsChangeModal
        isOpen={showConfirmModal}
        changes={pendingChanges}
        onConfirm={handleConfirmSave}
        onCancel={() => {
          setShowConfirmModal(false);
          setPendingChanges([]);
          setPendingProcessedValues(null);
        }}
        confirmLoading={saving}
      />
    </Card>
  );
};

export default DefaultUserSettings;
