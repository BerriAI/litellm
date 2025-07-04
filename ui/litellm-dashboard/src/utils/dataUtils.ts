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

export const formatNumberWithCommas = (value: number | null | undefined, decimals: number = 0): string => {
  if (value === null || value === undefined) {
    return '-';
  }
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};