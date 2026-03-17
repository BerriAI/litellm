import React, { useState, useEffect } from "react";
import { Card, Button, InputNumber, Typography, Spin, Select, Tag, Row, Col } from "antd";
import { EditOutlined, SaveOutlined } from "@ant-design/icons";
import { getDefaultTeamSettings, updateDefaultTeamSettings } from "./networking";
import BudgetDurationDropdown, { getBudgetDurationLabel } from "./common_components/budget_duration_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import NotificationsManager from "./molecules/notifications_manager";
import { ModelSelect } from "./ModelSelect/ModelSelect";

const { Title, Text } = Typography;

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
];

interface SettingRowProps {
  label: string;
  description: string;
  isEditing: boolean;
  viewContent: React.ReactNode;
  editContent: React.ReactNode;
}

const SettingRow: React.FC<SettingRowProps> = ({ label, description, isEditing, viewContent, editContent }) => (
  <Row className="py-5 border-b border-gray-100 last:border-0">
    <Col span={8} className="pr-6">
      <div className="text-sm font-semibold text-gray-900">{label}</div>
      <div className="text-xs text-gray-500 mt-1 leading-relaxed">{description}</div>
    </Col>
    <Col span={16} className="flex items-center">
      <div className="w-full">{isEditing ? editContent : viewContent}</div>
    </Col>
  </Row>
);

const NotSet = () => <Text className="text-gray-400 italic">Not set</Text>;

const renderTags = (values: string[], displayFn?: (v: string) => string) => {
  if (!values || values.length === 0) return <NotSet />;
  return (
    <div className="flex flex-wrap gap-2">
      {values.map((v) => (
        <Tag key={v} color="blue">
          {displayFn ? displayFn(v) : v}
        </Tag>
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
  const [editedValues, setEditedValues] = useState<SettingsValues>(DEFAULT_VALUES);
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
      const updatedSettings = await updateDefaultTeamSettings(accessToken, editedValues);
      const newValues = { ...DEFAULT_VALUES, ...(updatedSettings.settings || {}) };
      setValues(newValues);
      setEditedValues(newValues);
      setIsEditing(false);
      NotificationsManager.success("Default team settings updated successfully");
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

  const update = <K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) => {
    setEditedValues((prev) => ({ ...prev, [key]: value }));
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Spin size="large" />
      </div>
    );
  }

  if (fetchError) {
    return (
      <Card>
        <Text>No team settings available or you do not have permission to view them.</Text>
      </Card>
    );
  }

  return (
    <Card styles={{ body: { padding: 32 } }}>
      {/* Header */}
      <div className="flex justify-between items-start mb-2">
        <div>
          <Title level={3} className="m-0 text-gray-900">
            Default Team Settings
          </Title>
          <Text className="text-gray-500 mt-1 block">
            These settings will be applied by default when creating new teams.
          </Text>
        </div>
        <div>
          {isEditing ? (
            <div className="flex gap-3">
              <Button onClick={handleCancel} disabled={saving}>
                Cancel
              </Button>
              <Button type="primary" onClick={handleSave} loading={saving} icon={<SaveOutlined />}>
                Save Changes
              </Button>
            </div>
          ) : (
            <Button onClick={() => setIsEditing(true)} icon={<EditOutlined />}>
              Edit Settings
            </Button>
          )}
        </div>
      </div>

      <div className="mt-8">
        {/* Budget & Rate Limits */}
        <div className="mb-8">
          <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Budget & Rate Limits</div>
          <div className="border-t border-gray-100">
            <SettingRow
              label="Max Budget"
              description="Maximum budget (in USD) for new automatically created teams."
              isEditing={isEditing}
              viewContent={
                values.max_budget != null ? <Text>${Number(values.max_budget).toLocaleString()}</Text> : <NotSet />
              }
              editContent={
                <InputNumber
                  className="w-full"
                  style={{ maxWidth: 320 }}
                  value={editedValues.max_budget}
                  onChange={(v) => update("max_budget", v)}
                  placeholder="Not set"
                  prefix="$"
                  min={0}
                />
              }
            />

            <SettingRow
              label="Budget Duration"
              description="How frequently the team's budget resets."
              isEditing={isEditing}
              viewContent={
                values.budget_duration ? <Text>{getBudgetDurationLabel(values.budget_duration)}</Text> : <NotSet />
              }
              editContent={
                <BudgetDurationDropdown
                  value={editedValues.budget_duration || null}
                  onChange={(v) => update("budget_duration", v)}
                  style={{ maxWidth: 320 }}
                />
              }
            />

            <SettingRow
              label="TPM Limit"
              description="Maximum tokens per minute allowed across all models."
              isEditing={isEditing}
              viewContent={
                values.tpm_limit != null ? <Text>{values.tpm_limit.toLocaleString()}</Text> : <NotSet />
              }
              editContent={
                <InputNumber
                  className="w-full"
                  style={{ maxWidth: 320 }}
                  value={editedValues.tpm_limit}
                  onChange={(v) => update("tpm_limit", v)}
                  placeholder="Not set"
                  min={0}
                />
              }
            />

            <SettingRow
              label="RPM Limit"
              description="Maximum requests per minute allowed across all models."
              isEditing={isEditing}
              viewContent={
                values.rpm_limit != null ? <Text>{values.rpm_limit.toLocaleString()}</Text> : <NotSet />
              }
              editContent={
                <InputNumber
                  className="w-full"
                  style={{ maxWidth: 320 }}
                  value={editedValues.rpm_limit}
                  onChange={(v) => update("rpm_limit", v)}
                  placeholder="Not set"
                  min={0}
                />
              }
            />
          </div>
        </div>

        {/* Access & Permissions */}
        <div className="mb-8">
          <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Access & Permissions</div>
          <div className="border-t border-gray-100">
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
                <Select
                  mode="multiple"
                  style={{ width: "100%" }}
                  value={editedValues.team_member_permissions || []}
                  onChange={(v) => update("team_member_permissions", v)}
                  placeholder="Select permissions"
                  tagRender={({ label, closable, onClose }) => (
                    <Tag color="blue" closable={closable} onClose={onClose} className="mr-1 mt-1 mb-1">
                      {label}
                    </Tag>
                  )}
                >
                  {PERMISSION_OPTIONS.map((option) => (
                    <Select.Option key={option} value={option}>
                      {option}
                    </Select.Option>
                  ))}
                </Select>
              }
            />
          </div>
        </div>
      </div>
    </Card>
  );
};

export default TeamSSOSettings;
