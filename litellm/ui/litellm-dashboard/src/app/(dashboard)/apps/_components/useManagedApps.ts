import { useQuery } from "@tanstack/react-query";
import { appDefinitionKey, appResourcesKey, loadAppDefinition, loadAppResources, loadApps } from "./appApi";

export function useManagedApps(enabled = true) {
  const definition = useQuery({ queryKey: appDefinitionKey, queryFn: loadAppDefinition, enabled });
  const resourceType = definition.data?.resource_type ?? null;
  const resources = useQuery({
    queryKey: appResourcesKey(resourceType),
    queryFn: () => loadAppResources(resourceType!),
    enabled: enabled && resourceType !== null,
  });
  const apps = useQuery({
    queryKey: ["apps", resourceType],
    queryFn: () => loadApps(),
    enabled: enabled && resourceType !== null,
  });
  return { definition, resourceType, resources, apps };
}
