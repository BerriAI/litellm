import manifest from "../.github/issue-labels.json";

export const NAMESPACES = ["domain", "provider", "kind", "priority", "lift", "needs"] as const;
export type Namespace = (typeof NAMESPACES)[number];

export interface LabelSpec {
  readonly color: string;
  readonly description: string;
}

export type Manifest = Readonly<Record<Namespace, Readonly<Record<string, LabelSpec>>>>;

export interface ManifestLabel extends LabelSpec {
  readonly name: string;
}

export const MANIFEST: Manifest = manifest;

export function labelName(namespace: Namespace, value: string): string {
  return `${namespace}:${value}`;
}

export function namespaceOf(label: string): Namespace | undefined {
  const prefix = label.split(":")[0];
  return NAMESPACES.find((namespace) => namespace === prefix);
}

export function manifestLabels(source: Manifest): readonly ManifestLabel[] {
  return NAMESPACES.flatMap((namespace) =>
    Object.entries(source[namespace]).map(([value, spec]) => ({ name: labelName(namespace, value), ...spec })),
  );
}
