import React, { useState, useEffect, useCallback } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import {
  getCacheSettingsCall,
  testCacheConnectionCall,
  updateCacheSettingsCall,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import RedisTypeSelector from "./RedisTypeSelector";
import CacheFieldRenderer from "./CacheFieldRenderer";
import { gatherFormValues, groupFieldsByCategory } from "./cacheSettingsUtils";

interface CacheSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const CacheSettings: React.FC<CacheSettingsProps> = ({ accessToken }) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [cacheSettings, setCacheSettings] = useState<{ [key: string]: any }>({});
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [fields, setFields] = useState<any[]>([]);
  const [redisTypeDescriptions, setRedisTypeDescriptions] = useState<{
    [key: string]: string;
  }>({});
  const [redisType, setRedisType] = useState<string>("node");
  const [isTesting, setIsTesting] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  const loadCacheSettings = useCallback(async () => {
    try {
      const data = await getCacheSettingsCall(accessToken!);
      if (data.fields) setFields(data.fields);
      if (data.current_values) {
        setCacheSettings(data.current_values);
        if (data.current_values.redis_type) {
          setRedisType(data.current_values.redis_type);
        }
      }
      if (data.redis_type_descriptions) {
        setRedisTypeDescriptions(data.redis_type_descriptions);
      }
    } catch (error) {
      console.error("Failed to load cache settings:", error);
      NotificationsManager.fromBackend("Failed to load cache settings");
    }
  }, [accessToken]);

  useEffect(() => {
    if (!accessToken) return;
    loadCacheSettings();
  }, [accessToken, loadCacheSettings]);

  const handleTestConnection = async () => {
    if (!accessToken) return;
    setIsTesting(true);
    try {
      const testSettings = gatherFormValues(fields, redisType);
      const result = await testCacheConnectionCall(accessToken, testSettings);
      if (result.status === "success") {
        NotificationsManager.success("Cache connection test successful!");
      } else {
        NotificationsManager.fromBackend(
          `Connection test failed: ${result.message || result.error}`,
        );
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      console.error("Test connection error:", error);
      NotificationsManager.fromBackend(
        `Connection test failed: ${error.message || "Unknown error"}`,
      );
    } finally {
      setIsTesting(false);
    }
  };

  const handleSaveChanges = async () => {
    if (!accessToken) return;
    setIsSaving(true);
    try {
      const settingsToSave = gatherFormValues(fields, redisType);
      if (redisType === "semantic") settingsToSave.type = "redis-semantic";
      await updateCacheSettingsCall(accessToken, settingsToSave);
      NotificationsManager.success("Cache settings updated successfully");
      await loadCacheSettings();
    } catch (error) {
      console.error("Failed to save cache settings:", error);
      NotificationsManager.fromBackend("Failed to update cache settings");
    } finally {
      setIsSaving(false);
    }
  };

  if (!accessToken) return null;

  const {
    basicFields,
    sslFields,
    cacheManagementFields,
    gcpFields,
    clusterFields,
    sentinelFields,
    semanticFields,
  } = groupFieldsByCategory(fields, redisType);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const renderField = (field: any) => {
    if (!field) return null;
    const currentValue =
      cacheSettings[field.field_name] ?? field.field_default ?? "";
    return (
      <CacheFieldRenderer
        key={field.field_name}
        field={field}
        currentValue={currentValue}
      />
    );
  };

  return (
    <div className="w-full space-y-8 py-2">
      <div className="space-y-6">
        <div className="max-w-3xl">
          <h3 className="text-sm font-medium text-foreground">
            Cache Settings
          </h3>
          <p className="text-xs text-muted-foreground mt-1">
            Configure Redis cache for LiteLLM
          </p>
        </div>

        <RedisTypeSelector
          redisType={redisType}
          redisTypeDescriptions={redisTypeDescriptions}
          onTypeChange={setRedisType}
        />

        <div className="space-y-6 pt-4 border-t border-border">
          <h4 className="text-sm font-medium text-foreground">
            Connection Settings
          </h4>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {basicFields.map(renderField)}
          </div>
        </div>

        {redisType === "cluster" && clusterFields.length > 0 && (
          <div className="space-y-6 pt-4 border-t border-border">
            <h4 className="text-sm font-medium text-foreground">
              Cluster Configuration
            </h4>
            <div className="grid grid-cols-1 gap-6">
              {clusterFields.map(renderField)}
            </div>
          </div>
        )}

        {redisType === "sentinel" && sentinelFields.length > 0 && (
          <div className="space-y-6 pt-4 border-t border-border">
            <h4 className="text-sm font-medium text-foreground">
              Sentinel Configuration
            </h4>
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              {sentinelFields.map(renderField)}
            </div>
          </div>
        )}

        {redisType === "semantic" && semanticFields.length > 0 && (
          <div className="space-y-6 pt-4 border-t border-border">
            <h4 className="text-sm font-medium text-foreground">
              Semantic Configuration
            </h4>
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              {semanticFields.map(renderField)}
            </div>
          </div>
        )}

        <Accordion type="single" collapsible className="mt-4">
          <AccordionItem value="advanced">
            <AccordionTrigger className="text-sm font-medium">
              Advanced Settings
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-6">
                {sslFields.length > 0 && (
                  <div className="space-y-4">
                    <h5 className="text-sm font-medium text-foreground">
                      SSL Settings
                    </h5>
                    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                      {sslFields.map(renderField)}
                    </div>
                  </div>
                )}

                {cacheManagementFields.length > 0 && (
                  <div className="space-y-4 pt-4 border-t border-border">
                    <h5 className="text-sm font-medium text-foreground">
                      Cache Management
                    </h5>
                    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                      {cacheManagementFields.map(renderField)}
                    </div>
                  </div>
                )}

                {gcpFields.length > 0 && (
                  <div className="space-y-4 pt-4 border-t border-border">
                    <h5 className="text-sm font-medium text-foreground">
                      GCP Authentication
                    </h5>
                    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                      {gcpFields.map(renderField)}
                    </div>
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>

      <div className="border-t border-border pt-6 flex justify-end gap-3">
        <Button
          variant="secondary"
          size="sm"
          onClick={handleTestConnection}
          disabled={isTesting}
        >
          {isTesting ? "Testing..." : "Test Connection"}
        </Button>
        <Button size="sm" onClick={handleSaveChanges} disabled={isSaving}>
          {isSaving ? "Saving..." : "Save Changes"}
        </Button>
      </div>
    </div>
  );
};

export default CacheSettings;
