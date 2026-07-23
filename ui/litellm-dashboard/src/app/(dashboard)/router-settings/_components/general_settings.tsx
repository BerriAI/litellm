import React, { useState, useEffect } from "react";
import {
  Card,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  TableBody,
  Title,
  Text,
  Button,
  Icon,
  Switch,
} from "@tremor/react";
import { getGeneralSettingsCall, updateConfigFieldSetting, deleteConfigFieldSetting } from "@/components/networking";
import { InputNumber, Select as AntdSelect } from "antd";
import { TrashIcon } from "@heroicons/react/outline";
import { StatusBadge } from "@/components/shared/table_cells";

const PROMPT_CACHING_TAB = "prompt_caching";
const ENABLE_ANTHROPIC_PROMPT_CACHING = "enable_anthropic_prompt_caching";
const ANTHROPIC_PROMPT_CACHING_TTL = "anthropic_prompt_caching_ttl";

export interface generalSettingsItem {
  field_name: string;
  field_type: string;
  field_value: any;
  field_description: string;
  stored_in_db: boolean | null;
  field_options?: string[] | null;
  field_tab?: string | null;
  field_default_value?: any;
}

const SettingValueEditor: React.FC<{
  setting: generalSettingsItem;
  onChange: (fieldName: string, newValue: any) => void;
}> = ({ setting, onChange }) => {
  if (setting.field_type === "Integer") {
    return (
      <InputNumber
        step={1}
        value={setting.field_value}
        onChange={(newValue) => onChange(setting.field_name, newValue)}
      />
    );
  }
  if (setting.field_type === "Boolean") {
    return (
      <Switch
        checked={setting.field_value === true || setting.field_value === "true"}
        onChange={(checked) => onChange(setting.field_name, checked)}
      />
    );
  }
  if (setting.field_type === "Float") {
    return (
      <InputNumber
        min={0}
        max={1}
        step={0.05}
        value={setting.field_value}
        onChange={(newValue) => onChange(setting.field_name, newValue)}
      />
    );
  }
  if (setting.field_type === "Dollar") {
    return (
      <InputNumber
        min={0.01}
        step={0.25}
        prefix="$"
        value={setting.field_value}
        onChange={(newValue) => onChange(setting.field_name, newValue)}
      />
    );
  }
  if (setting.field_type === "Select") {
    return (
      <AntdSelect
        allowClear
        style={{ minWidth: "8rem" }}
        placeholder="Default"
        value={setting.field_value || undefined}
        options={(setting.field_options ?? []).map((option) => ({ label: option, value: option }))}
        onChange={(newValue) => onChange(setting.field_name, newValue ?? "")}
      />
    );
  }
  return null;
};

export const PromptCachingPanel: React.FC<{
  accessToken: string;
  settings: generalSettingsItem[];
  onChange: (fieldName: string, newValue: any) => void;
}> = ({ accessToken, settings, onChange }) => {
  const enableSetting = settings.find((s) => s.field_name === ENABLE_ANTHROPIC_PROMPT_CACHING);
  const ttlSetting = settings.find((s) => s.field_name === ANTHROPIC_PROMPT_CACHING_TTL);

  // The two rows come from the same registry the General tab reads; if they
  // are not loaded yet there is nothing to render.
  if (!enableSetting) {
    return null;
  }

  const enabled = enableSetting.field_value === true || enableSetting.field_value === "true";

  // Apply immediately: a toggle and a dropdown are direct controls, so there is
  // no separate Update button. Clearing the ttl resets it to the provider default.
  const persist = (fieldName: string, value: any) => {
    onChange(fieldName, value);
    if (value === "" || value === null || value === undefined) {
      deleteConfigFieldSetting(accessToken, fieldName);
    } else {
      updateConfigFieldSetting(accessToken, fieldName, value);
    }
  };

  return (
    <Card>
      <Title>Prompt Caching</Title>

      <div className="mt-6 flex items-start justify-between gap-8">
        <div className="max-w-2xl">
          <Text className="font-medium">Automatic Anthropic prompt caching</Text>
          <p className="mt-1 text-xs text-gray-500">{enableSetting.field_description}</p>
        </div>
        <Switch checked={enabled} onChange={(checked) => persist(ENABLE_ANTHROPIC_PROMPT_CACHING, checked)} />
      </div>

      {ttlSetting && (
        <div className="mt-6 flex items-start justify-between gap-8">
          <div className="max-w-2xl">
            <Text className={`font-medium ${enabled ? "" : "text-gray-400"}`}>Cache lifetime (TTL)</Text>
            <p className="mt-1 text-xs text-gray-500">{ttlSetting.field_description}</p>
          </div>
          <AntdSelect
            allowClear
            disabled={!enabled}
            style={{ minWidth: "10rem" }}
            placeholder="5m (default)"
            value={ttlSetting.field_value || undefined}
            options={(ttlSetting.field_options ?? []).map((option) => ({ label: option, value: option }))}
            onChange={(newValue) => persist(ANTHROPIC_PROMPT_CACHING_TTL, newValue ?? "")}
          />
        </div>
      )}
    </Card>
  );
};

const useGeneralSettings = (accessToken: string) => {
  const [generalSettings, setGeneralSettings] = useState<generalSettingsItem[]>([]);

  useEffect(() => {
    getGeneralSettingsCall(accessToken).then((data) => {
      setGeneralSettings(data);
    });
  }, [accessToken]);

  const handleInputChange = (fieldName: string, newValue: any) => {
    setGeneralSettings((prev) =>
      prev.map((setting) => (setting.field_name === fieldName ? { ...setting, field_value: newValue } : setting)),
    );
  };

  const handleUpdateField = (fieldName: string) => {
    const fieldValue = generalSettings.find((setting) => setting.field_name === fieldName)?.field_value;
    if (fieldValue == null) {
      return;
    }
    updateConfigFieldSetting(accessToken, fieldName, fieldValue);
    setGeneralSettings((prev) =>
      prev.map((setting) => (setting.field_name === fieldName ? { ...setting, stored_in_db: true } : setting)),
    );
  };

  const handleResetField = (fieldName: string) => {
    deleteConfigFieldSetting(accessToken, fieldName);
    setGeneralSettings((prev) =>
      prev.map((setting) =>
        setting.field_name === fieldName
          ? { ...setting, stored_in_db: null, field_value: setting.field_default_value ?? null }
          : setting,
      ),
    );
  };

  return { generalSettings, handleInputChange, handleUpdateField, handleResetField };
};

export const PromptCachingSettingsTab: React.FC<{ accessToken: string }> = ({ accessToken }) => {
  const { generalSettings, handleInputChange } = useGeneralSettings(accessToken);
  return <PromptCachingPanel accessToken={accessToken} settings={generalSettings} onChange={handleInputChange} />;
};

export const GeneralConfigTab: React.FC<{ accessToken: string }> = ({ accessToken }) => {
  const { generalSettings, handleInputChange, handleUpdateField, handleResetField } = useGeneralSettings(accessToken);

  return (
    <Card>
      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>Setting</TableHeaderCell>
            <TableHeaderCell>Value</TableHeaderCell>
            <TableHeaderCell>Status</TableHeaderCell>
            <TableHeaderCell>Action</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {generalSettings
            .filter((value) => value.field_type !== "TypedDictionary" && value.field_tab !== PROMPT_CACHING_TAB)
            .map((value, index) => (
              <TableRow key={index}>
                <TableCell>
                  <Text>{value.field_name}</Text>
                  <p
                    style={{
                      fontSize: "0.65rem",
                      color: "#808080",
                      fontStyle: "italic",
                    }}
                    className="mt-1"
                  >
                    {value.field_description}
                  </p>
                </TableCell>
                <TableCell>
                  <SettingValueEditor setting={value} onChange={handleInputChange} />
                </TableCell>
                <TableCell>
                  {value.stored_in_db == true ? (
                    <StatusBadge tone="success" label="In DB" />
                  ) : value.stored_in_db == false ? (
                    <StatusBadge tone="neutral" label="In Config" />
                  ) : (
                    <StatusBadge tone="neutral" label="Not Set" />
                  )}
                </TableCell>
                <TableCell>
                  <Button onClick={() => handleUpdateField(value.field_name)}>Update</Button>
                  <Icon icon={TrashIcon} color="red" onClick={() => handleResetField(value.field_name)}>
                    Reset
                  </Icon>
                </TableCell>
              </TableRow>
            ))}
        </TableBody>
      </Table>
    </Card>
  );
};
