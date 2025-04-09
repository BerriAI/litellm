export function updateExistingKeys<Source extends Object>(
  target: Source,
  source: Object
): Source {
  const clonedTarget = structuredClone(target);
  
  for (const [key, value] of Object.entries(source)) {
    if (key in clonedTarget) {
      (clonedTarget as any)[key] = value;
    }
  }

  return clonedTarget;
}
