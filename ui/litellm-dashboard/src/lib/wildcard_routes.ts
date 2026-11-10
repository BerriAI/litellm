export type WildcardDeployment = {
  readonly modelGroup: string;
  readonly underlyingModels: readonly string[];
  readonly blocked?: boolean;
};
export type WildcardRoute = { readonly model: string; readonly sourceGroup: string; readonly paused: boolean };

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

const partialWordBeforeStar = (pattern: string): string =>
  pattern.slice(0, pattern.indexOf("*")).split("/").at(-1) ?? "";

const routesCapturedIds = (namePattern: string, deploymentPattern: string): boolean => {
  const singleStar = namePattern.split("*").length === 2 && deploymentPattern.split("*").length === 2;
  const listsCatalogIds = isProviderWide(namePattern) && isProviderWide(deploymentPattern);
  if (!deploymentPattern.includes("/") || listsCatalogIds) return false;
  const listsNameFragments =
    /[a-z0-9]$/i.test(partialWordBeforeStar(deploymentPattern)) && partialWordBeforeStar(namePattern) === "";
  return singleStar && !listsNameFragments;
};

const substituteCapture = (namePattern: string, deploymentPattern: string, group: string): string => {
  const [head, tail] = namePattern.split("*");
  return deploymentPattern.replace("*", () => group.slice(head.length, group.length - tail.length));
};

const routingPatterns = (deployment: WildcardDeployment): readonly string[] =>
  deployment.modelGroup === "*"
    ? deployment.underlyingModels.filter((model) => model !== "*" && isProviderWide(model))
    : [deployment.modelGroup].filter((pattern) => pattern.includes("*"));

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
  return models.map((model) => ({ model, sourceGroup: deployment.modelGroup, paused: Boolean(deployment.blocked) }));
};

const PYTHON_ESCAPED_CHARS = new Set(Array.from("()[]{}?*+-|^$\\.&~# \t\n\r\v\f"));
const ROUTER_COMPLEXITY_CHARS = new Set(Array.from("*+?\\^$|()"));

const routerRegexPart = (char: string): string => {
  if (char === "*") return "(.*)";
  return PYTHON_ESCAPED_CHARS.has(char) ? `\\${char}` : char;
};

const routerSpecificity = (namePattern: string): readonly [number, number] => {
  const regex = Array.from(namePattern, routerRegexPart).join("");
  return [regex.length, Array.from(regex).filter((char) => ROUTER_COMPLEXITY_CHARS.has(char)).length];
};

const isMoreSpecific = (namePattern: string, than: string): boolean => {
  const [length, complexity] = routerSpecificity(namePattern);
  const [otherLength, otherComplexity] = routerSpecificity(than);
  return length > otherLength || (length === otherLength && complexity > otherComplexity);
};

const catchAllRoutes = (group: string, deployments: readonly WildcardDeployment[]): readonly WildcardRoute[] =>
  deployments
    .filter((deployment) => deployment.modelGroup === "*")
    .flatMap((deployment) =>
      deployment.underlyingModels
        .filter((model) => routesCapturedIds("*", model))
        .map((model) => ({
          model: substituteCapture("*", model, group),
          sourceGroup: "*",
          paused: Boolean(deployment.blocked),
        })),
    );

const routesByPrecedence = (group: string, deployments: readonly WildcardDeployment[]): readonly WildcardRoute[] => {
  const matches = deployments.flatMap((deployment) =>
    routingPatterns(deployment)
      .filter((pattern) => matchesWildcard(pattern, group))
      .map((pattern) => ({ deployment, pattern })),
  );
  if (matches.length === 0) return catchAllRoutes(group, deployments);
  const routedName = matches
    .map(({ deployment }) => deployment.modelGroup)
    .reduce((best, name) => (isMoreSpecific(name, best) ? name : best));
  return matches
    .filter(({ deployment }) => deployment.modelGroup === routedName)
    .flatMap(({ deployment, pattern }) => routesThroughPattern(deployment, pattern, group));
};

export const wildcardRoutes = (
  group: string,
  deployments: readonly WildcardDeployment[],
  groupProviders: readonly string[],
): readonly WildcardRoute[] =>
  routesByPrecedence(group, deployments).filter(
    ({ model }) => model === group || groupProviders.includes(model.split("/")[0]),
  );
