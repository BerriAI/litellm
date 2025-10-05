import { mapDisplayToInternalNames } from "@/components/callback_info_helpers";
import { ModelAliases } from "@/app/(console)/virtual-keys/components/CreateKeyModal/types";

export const getPredefinedTags = (data: any[] | null) => {
  let allTags = [];

  console.log("data:", JSON.stringify(data));

  if (data) {
    for (let key of data) {
      if (key["metadata"] && key["metadata"]["tags"]) {
        allTags.push(...key["metadata"]["tags"]);
      }
    }
  }

  const uniqueTags = Array.from(new Set(allTags)).map((tag) => ({
    value: tag,
    label: tag,
  }));

  console.log("uniqueTags:", uniqueTags);
  return uniqueTags;
};

type AnyObj = Record<string, any>;

type PrepareOptions = {
  keyOwner: string;
  userID: string;
  loggingSettings: any[];
  disabledCallbacks: string[];
  autoRotationEnabled: boolean;
  rotationInterval: string;
  modelAliases: ModelAliases;
};

export function safeParseMetadata(metadataStr: string | undefined): AnyObj {
  try {
    return JSON.parse(metadataStr || "{}");
  } catch (error) {
    console.error("Error parsing metadata:", error);
    return {};
  }
}

export function withOwnerAssignments(
  values: AnyObj,
  { keyOwner, userID }: Pick<PrepareOptions, "keyOwner" | "userID">,
): AnyObj {
  // If owned by "you", set user_id
  const base = keyOwner === "you" ? { ...values, user_id: userID } : { ...values };
  return base;
}

export function enrichMetadata(
  metadata: AnyObj,
  values: AnyObj,
  {
    keyOwner,
    loggingSettings,
    disabledCallbacks,
  }: Pick<PrepareOptions, "keyOwner" | "loggingSettings" | "disabledCallbacks">,
): AnyObj {
  let next = { ...metadata };

  // If it's a service account, add the service_account_id to the metadata
  if (keyOwner === "service_account") {
    next = { ...next, service_account_id: values.key_alias };
  }

  // Add logging settings to the metadata
  if (Array.isArray(loggingSettings) && loggingSettings.length > 0) {
    next = { ...next, logging: loggingSettings.filter((config: any) => config?.callback_name) };
  }

  // Add disabled callbacks to the metadata
  if (Array.isArray(disabledCallbacks) && disabledCallbacks.length > 0) {
    const mappedDisabledCallbacks = mapDisplayToInternalNames(disabledCallbacks);
    next = { ...next, litellm_disabled_callbacks: mappedDisabledCallbacks };
  }

  return next;
}

export function withAutoRotation(
  values: AnyObj,
  { autoRotationEnabled, rotationInterval }: Pick<PrepareOptions, "autoRotationEnabled" | "rotationInterval">,
): AnyObj {
  if (!autoRotationEnabled) return { ...values };
  return { ...values, auto_rotate: true, rotation_interval: rotationInterval };
}

export function withDurationNoop(values: AnyObj): AnyObj {
  // Preserve the original no-op logic: if duration exists, set it to itself
  if (values.duration) {
    return { ...values, duration: values.duration };
  }
  return { ...values };
}

export function mergeObjectPermission(values: AnyObj, patch: AnyObj): AnyObj {
  const object_permission = { ...(values.object_permission || {}), ...patch };
  return { ...values, object_permission };
}

export function withVectorStores(values: AnyObj): AnyObj {
  const { allowed_vector_store_ids, ...rest } = values;
  if (Array.isArray(allowed_vector_store_ids) && allowed_vector_store_ids.length > 0) {
    const patched = mergeObjectPermission(rest, { vector_stores: allowed_vector_store_ids });
    return patched; // original field removed by destructuring
  }
  return { ...values };
}

export function withMcpServersAndGroups(values: AnyObj): AnyObj {
  const { allowed_mcp_servers_and_groups, ...rest } = values;
  const servers = allowed_mcp_servers_and_groups?.servers;
  const accessGroups = allowed_mcp_servers_and_groups?.accessGroups;

  const hasServers = Array.isArray(servers) && servers.length > 0;
  const hasAccessGroups = Array.isArray(accessGroups) && accessGroups.length > 0;

  if (!hasServers && !hasAccessGroups) return { ...values };

  let patched = { ...rest };
  if (hasServers) patched = mergeObjectPermission(patched, { mcp_servers: servers });
  if (hasAccessGroups) patched = mergeObjectPermission(patched, { mcp_access_groups: accessGroups });

  // original field removed by destructuring
  return patched;
}

export function withMcpAccessGroups(values: AnyObj): AnyObj {
  const { allowed_mcp_access_groups, ...rest } = values;
  if (Array.isArray(allowed_mcp_access_groups) && allowed_mcp_access_groups.length > 0) {
    const patched = mergeObjectPermission(rest, { mcp_access_groups: allowed_mcp_access_groups });
    return patched; // original field removed by destructuring
  }
  return { ...values };
}

function withAliases(values: AnyObj, modelAliases: ModelAliases): AnyObj {
  if (modelAliases && Object.keys(modelAliases).length > 0) {
    return { ...values, aliases: JSON.stringify(modelAliases) };
  }
  return { ...values };
}

function withMetadataString(values: AnyObj, metadata: AnyObj): AnyObj {
  return { ...values, metadata: JSON.stringify(metadata) };
}

export function prepareFormValues(initialValues: AnyObj, opts: PrepareOptions): AnyObj {
  // Step 1: assign user id if needed
  let values = withOwnerAssignments(initialValues, { keyOwner: opts.keyOwner, userID: opts.userID });

  // Step 2: metadata parse & enrichment
  const parsedMetadata = safeParseMetadata(values.metadata);
  const enrichedMetadata = enrichMetadata(parsedMetadata, values, {
    keyOwner: opts.keyOwner,
    loggingSettings: opts.loggingSettings,
    disabledCallbacks: opts.disabledCallbacks,
  });

  // Step 3: auto-rotation
  values = withAutoRotation(values, {
    autoRotationEnabled: opts.autoRotationEnabled,
    rotationInterval: opts.rotationInterval,
  });

  // Step 4: duration (no-op preservation)
  values = withDurationNoop(values);

  // Step 5: commit metadata string
  values = withMetadataString(values, enrichedMetadata);

  // Step 6: object_permission transformations
  values = withVectorStores(values);
  values = withMcpServersAndGroups(values);
  values = withMcpAccessGroups(values);

  // Step 7: aliases
  values = withAliases(values, opts.modelAliases);

  return values;
}
