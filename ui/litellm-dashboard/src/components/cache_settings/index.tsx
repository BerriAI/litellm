import React, { useState, useEffect } from "react";
import { Button } from "@tremor/react";
import {
  getCacheSettingsCall,
  testCacheConnectionCall,
  updateCacheSettingsCall,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import RedisTypeSelector from "./RedisTypeSelector";
import CacheFieldGroup from "./CacheFieldGroup";

interface CacheSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const CacheSettings: React.FC<CacheSettingsProps> = ({ accessToken, userRole, userID }) => {
  const [cacheSettings, setCacheSettings] = useState<{ [key: string]: any }>({});
  const [fields, setFields] = useState<any[]>([]);
  const [redisTypeDescriptions, setRedisTypeDescriptions] = useState<{ [key: string]: string }>({});
  const [redisType, setRedisType] = useState<string>("node");
  const [isTesting, setIsTesting] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    loadCacheSettings();
  }, [accessToken]);

  const loadCacheSettings = async () => {
    try {
      const data = await getCacheSettingsCall(accessToken!);
      console.log("cache settings from API", data);
      
      if (data.fields) {
        setFields(data.fields);
      }

      // Set current values
      if (data.current_values) {
        setCacheSettings(data.current_values);
        if (data.current_values.redis_type) {
          setRedisType(data.current_values.redis_type);
        }
      }

      // Store Redis type descriptions
      if (data.redis_type_descriptions) {
        setRedisTypeDescriptions(data.redis_type_descriptions);
      }
    } catch (error) {
      console.error("Failed to load cache settings:", error);
      NotificationsManager.fromBackend("Failed to load cache settings");
    }
  };

  const handleTestConnection = async () => {
    if (!accessToken) {
      return;
    }

    setIsTesting(true);
    try {
      const testSettings = gatherFormValues();
      const result = await testCacheConnectionCall(accessToken, testSettings);
      
      if (result.status === "success") {
        NotificationsManager.success("Cache connection test successful!");
      } else {
        NotificationsManager.fromBackend(`Connection test failed: ${result.message || result.error}`);
      }
    } catch (error: any) {
      console.error("Test connection error:", error);
      NotificationsManager.fromBackend(`Connection test failed: ${error.message || "Unknown error"}`);
    } finally {
      setIsTesting(false);
    }
  };

  const handleSaveChanges = async () => {
    if (!accessToken) {
      return;
    }

    setIsSaving(true);
    try {
      const settingsToSave = gatherFormValues();
      await updateCacheSettingsCall(accessToken, settingsToSave);
      NotificationsManager.success("Cache settings updated successfully");
      // Reload settings to reflect saved values
      await loadCacheSettings();
    } catch (error) {
      console.error("Failed to save cache settings:", error);
      NotificationsManager.fromBackend("Failed to update cache settings");
    } finally {
      setIsSaving(false);
    }
  };

  const gatherFormValues = (): { [key: string]: any } => {
    const values: { [key: string]: any } = {
      redis_type: redisType,
    };

    // Iterate through all fields from backend
    fields.forEach((field) => {
      // Skip redis_type as we handle it separately
      if (field.field_name === "redis_type") {
        return;
      }

      // Check if field should be shown for current redis type
      if (!shouldShowField(field)) {
        return;
      }

      const fieldName = field.field_name;
      let value: any = null;

      if (field.field_type === "Boolean") {
        const checkboxEl = document.querySelector(`input[name="${fieldName}"]`) as HTMLInputElement | null;
        if (checkboxEl?.checked !== undefined) {
          value = checkboxEl.checked;
        }
      } else if (field.field_type === "List") {
        const textareaEl = document.querySelector(`textarea[name="${fieldName}"]`) as HTMLTextAreaElement | null;
        if (textareaEl?.value) {
          try {
            value = JSON.parse(textareaEl.value);
          } catch (e) {
            console.error(`Invalid JSON for ${fieldName}:`, e);
          }
        }
      } else {
        const inputEl = document.querySelector(`input[name="${fieldName}"]`) as HTMLInputElement | null;
        if (inputEl?.value) {
          const trimmedValue = inputEl.value.trim();
          if (trimmedValue !== "") {
            if (field.field_type === "Integer") {
              const num = Number(trimmedValue);
              if (!isNaN(num)) value = num;
            } else if (field.field_type === "Float") {
              const num = Number(trimmedValue);
              if (!isNaN(num)) value = num;
            } else {
              value = trimmedValue;
            }
          }
        }
      }

      if (value !== null && value !== undefined) {
        values[fieldName] = value;
      }
    });

    return values;
  };

  const shouldShowField = (field: any): boolean => {
    // Show field if it applies to all types (redis_type is null/undefined) or to current selected type
    if (field.redis_type === null || field.redis_type === undefined) {
      return true;
    }
    
    return field.redis_type === redisType;
  };

  if (!accessToken) {
    return null;
  }

  // Group fields by redis_type for better organization
  const commonFields = fields.filter(f => !f.redis_type || f.redis_type === null);
  const clusterFields = fields.filter(f => f.redis_type === "cluster");
  const sentinelFields = fields.filter(f => f.redis_type === "sentinel");
  const gcpFields = fields.filter(f => 
    f.field_name === "gcp_service_account" || 
    f.field_name === "gcp_ssl_ca_certs" || 
    f.field_name === "ssl_cert_reqs" || 
    f.field_name === "ssl_check_hostname"
  );

  return (
    <div className="w-full space-y-8 py-2">
      <div className="space-y-6">
        <div className="max-w-3xl">
          <h3 className="text-sm font-medium text-gray-900">Cache Settings</h3>
          <p className="text-xs text-gray-500 mt-1">Configure Redis cache for LiteLLM</p>
        </div>

        {/* Redis Type Selector */}
        <RedisTypeSelector
          redisType={redisType}
          redisTypeDescriptions={redisTypeDescriptions}
          onTypeChange={setRedisType}
        />

        {/* Common Fields */}
        <CacheFieldGroup
          title="Configuration"
          fields={commonFields}
          cacheSettings={cacheSettings}
          redisType={redisType}
        />

        {/* Cluster-specific Fields */}
        {redisType === "cluster" && (
          <CacheFieldGroup
            title="Cluster Configuration"
            fields={clusterFields}
            cacheSettings={cacheSettings}
            redisType={redisType}
            gridCols="grid-cols-1 gap-6"
          />
        )}

        {/* Sentinel-specific Fields */}
        {redisType === "sentinel" && (
          <CacheFieldGroup
            title="Sentinel Configuration"
            fields={sentinelFields}
            cacheSettings={cacheSettings}
            redisType={redisType}
          />
        )}

        {/* GCP IAM Fields */}
        <CacheFieldGroup
          title="GCP IAM Authentication (Optional)"
          fields={gcpFields}
          cacheSettings={cacheSettings}
          redisType={redisType}
        />
      </div>

      {/* Actions */}
      <div className="border-t border-gray-200 pt-6 flex justify-end gap-3">
        <Button
          variant="secondary"
          size="sm"
          onClick={handleTestConnection}
          disabled={isTesting}
          className="text-sm"
        >
          {isTesting ? "Testing..." : "Test Connection"}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => window.location.reload()}
          className="text-sm"
        >
          Reset
        </Button>
        <Button
          size="sm"
          onClick={handleSaveChanges}
          disabled={isSaving}
          className="text-sm font-medium"
        >
          {isSaving ? "Saving..." : "Save Changes"}
        </Button>
      </div>
    </div>
  );
};

export default CacheSettings;
