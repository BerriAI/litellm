export type WildcardDeployment = { readonly modelGroup: string; readonly underlyingModels: readonly string[] };
export type WildcardRoute = { readonly model: string; readonly sourceGroup: string };

// A linear glob scan rather than a RegExp: patterns are admin-controlled model_name values, and a
// backtracking regex built from one ("a*a*a*...") can freeze another admin's dashboard.
const matchesWildcard = (pattern: string, name: string): boolean => {
  const parts = pattern.split("*");
  const head = parts[0];
  const tail = parts[parts.length - 1];
  if (!name.startsWith(head) || !name.endsWith(tail)) return false;
  if (name.length < head.length + tail.length) return false;
  const scanEnd = name.length - tail.length;
  const scanResult = parts.slice(1, -1).reduce((searchFrom: number, part: string) => {
    if (searchFrom < 0) return -1;
    const found = name.indexOf(part, searchFrom);
    return found === -1 || found + part.length > scanEnd ? -1 : found + part.length;
  }, head.length);
  return scanResult >= 0;
};

const isProviderWide = (pattern: string): boolean => pattern.slice(pattern.indexOf("/") + 1) === "*";

const routesCapturedIds = (namePattern: string, deploymentPattern: string): boolean => {
  const singleStar = namePattern.split("*").length === 2 && deploymentPattern.split("*").length === 2;
  const partialWord = deploymentPattern.slice(0, deploymentPattern.indexOf("*")).split("/").at(-1) ?? "";
  const listsCatalogIds = isProviderWide(namePattern) && isProviderWide(deploymentPattern);
  if (!deploymentPattern.includes("/") || listsCatalogIds) return false;
  return singleStar && !/[a-z0-9]$/i.test(partialWord);
};

const substituteCapture = (namePattern: string, deploymentPattern: string, group: string): string => {
  const [head, tail] = namePattern.split("*");
  return deploymentPattern.replace("*", () => group.slice(head.length, group.length - tail.length));
};

const listingPatterns = (deployment: WildcardDeployment): readonly string[] =>
  (deployment.modelGroup === "*" ? deployment.underlyingModels.filter(isProviderWide) : [deployment.modelGroup]).filter(
    (pattern) => pattern !== "*" && pattern.includes("*") && pattern.includes("/"),
  );

const routesThroughPattern = (
  deployment: WildcardDeployment,
  pattern: string,
  group: string,
): readonly WildcardRoute[] => {
  const capturing =
    deployment.modelGroup === "*"
      ? []
      : deployment.underlyingModels.filter((model) => routesCapturedIds(pattern, model));
  const models = capturing.length > 0 ? capturing.map((model) => substituteCapture(pattern, model, group)) : [group];
  return models.map((model) => ({ model, sourceGroup: deployment.modelGroup }));
};

export const wildcardRoutes = (group: string, deployments: readonly WildcardDeployment[]): readonly WildcardRoute[] => {
  const listed = deployments.flatMap((deployment) =>
    listingPatterns(deployment)
      .filter((pattern) => matchesWildcard(pattern, group))
      .flatMap((pattern) => routesThroughPattern(deployment, pattern, group)),
  );
  if (listed.length > 0) return listed;
  return deployments
    .filter((deployment) => deployment.modelGroup === "*")
    .flatMap((deployment) => deployment.underlyingModels.filter((model) => routesCapturedIds("*", model)))
    .map((model) => ({ model: substituteCapture("*", model, group), sourceGroup: "*" }));
};
