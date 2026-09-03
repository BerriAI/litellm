import { InputSchemaProperty } from "./types";

// JSON Schema allows "type" to be an array (e.g. ["string", "null"]) for a nullable field, the
// shape Pydantic's model_json_schema() emits for Optional[str] / str | None. Every branch that
// decides a widget or a value conversion from a property's type needs the single effective
// (non-null) type, not the raw field verbatim.
export function resolveSchemaType(type: InputSchemaProperty["type"] | undefined): string | undefined {
  if (Array.isArray(type)) {
    return type.find((t) => t !== "null") ?? type[0];
  }
  return type;
}

const isPlainObject = (value: unknown): value is Record<string, any> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function buildArrayItems(items?: InputSchemaProperty | InputSchemaProperty[]): any[] {
  if (!items) return [];
  if (Array.isArray(items)) {
    return items.map((item) => buildDefaultValue(item)).filter((value) => value !== undefined);
  }
  const itemDefault = buildDefaultValue(items);
  return itemDefault !== undefined ? [itemDefault] : [];
}

export function buildDefaultValue(prop?: InputSchemaProperty, overrideDefault?: any): any {
  if (!prop) return undefined;
  const effectiveDefault = overrideDefault !== undefined ? overrideDefault : prop.default;
  const effectiveType = resolveSchemaType(prop.type);

  if (effectiveType === "object") {
    const base = isPlainObject(effectiveDefault) ? { ...effectiveDefault } : {};
    if (prop.properties) {
      Object.entries(prop.properties).forEach(([childKey, childProp]) => {
        base[childKey] = buildDefaultValue(childProp, base[childKey]);
      });
    }
    return base;
  }

  if (effectiveType === "array") {
    if (Array.isArray(effectiveDefault)) {
      const itemSchema = prop.items;
      if (!itemSchema) return effectiveDefault;
      if (effectiveDefault.length === 0) {
        const sample = buildArrayItems(itemSchema);
        return sample.length ? sample : effectiveDefault;
      }
      if (Array.isArray(itemSchema)) {
        return effectiveDefault.map((value, index) => {
          const schema = itemSchema[index] ?? itemSchema[itemSchema.length - 1];
          return buildDefaultValue(schema, value);
        });
      }
      return effectiveDefault.map((value) => buildDefaultValue(itemSchema, value));
    }
    if (effectiveDefault !== undefined) return effectiveDefault;
    return buildArrayItems(prop.items);
  }

  if (effectiveDefault !== undefined) return effectiveDefault;

  // A nullable numeric/boolean field (an array "type") with no explicit default starts empty
  // rather than a synthetic 0/false: unlike a string's natural "" default, that value looks
  // user-provided, passes the non-empty submission filter, and gets sent to the tool even when
  // the field was never touched.
  const isNullable = Array.isArray(prop.type);
  switch (effectiveType) {
    case "integer":
    case "number":
      return isNullable ? undefined : 0;
    case "boolean":
      return isNullable ? undefined : false;
    case "string":
    default:
      return "";
  }
}

export const getInitialValueForField = (prop: InputSchemaProperty): any => {
  const defaultValue = buildDefaultValue(prop);
  const effectiveType = resolveSchemaType(prop.type);
  if (effectiveType === "object" || effectiveType === "array") {
    const fallback = effectiveType === "array" ? [] : {};
    return JSON.stringify(defaultValue ?? fallback, null, 2);
  }
  return defaultValue;
};
