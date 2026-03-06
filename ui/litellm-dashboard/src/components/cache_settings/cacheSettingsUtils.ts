/**
 * Utility functions for cache settings form handling
 */

export const shouldShowField = (field: any, redisType: string): boolean => {
  // Show field if it applies to all types (redis_type is null/undefined) or to current selected type
  if (field.redis_type === null || field.redis_type === undefined) {
    return true;
  }

  return field.redis_type === redisType;
};

export const getFieldByName = (fields: any[], fieldName: string) => {
  return fields.find((f) => f.field_name === fieldName);
};

export const groupFieldsByCategory = (fields: any[], redisType: string) => {
  // Basic fields that are always shown
  const basicFieldNames = ["host", "port", "password", "username"];
  const basicFields = basicFieldNames.map((name) => getFieldByName(fields, name)).filter(Boolean);

  // Advanced field groups
  const sslFields = ["ssl", "ssl_cert_reqs", "ssl_check_hostname"]
    .map((name) => getFieldByName(fields, name))
    .filter(Boolean);

  const cacheManagementFields = ["namespace", "ttl", "max_connections"]
    .map((name) => getFieldByName(fields, name))
    .filter(Boolean);

  const gcpFields = ["gcp_service_account", "gcp_ssl_ca_certs"]
    .map((name) => getFieldByName(fields, name))
    .filter(Boolean);

  // Redis type-specific fields
  const clusterFields = fields.filter((f) => f.redis_type === "cluster");
  const sentinelFields = fields.filter((f) => f.redis_type === "sentinel");
  const semanticFields = fields.filter((f) => f.redis_type === "semantic");

  return {
    basicFields,
    sslFields,
    cacheManagementFields,
    gcpFields,
    clusterFields,
    sentinelFields,
    semanticFields,
  };
};

export const gatherFormValues = (fields: any[], redisType: string): { [key: string]: any } => {
  const values: { [key: string]: any } = {
    type: "redis", // Cache class accepts 'type' parameter (LiteLLMCacheType enum)
  };

  // Iterate through all fields from backend
  fields.forEach((field) => {
    // Skip redis_type - it's UI-only, not sent to backend
    if (field.field_name === "redis_type") {
      return;
    }

    // Check if field should be shown for current redis type
    if (!shouldShowField(field, redisType)) {
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
