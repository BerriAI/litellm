import { message } from "antd";

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

export const copyToClipboard = async (
  text: string | null | undefined,
  messageText: string = "Copied to clipboard"
): Promise<boolean> => {
  if (!text) return false;
  try {
    await navigator.clipboard.writeText(text);
    message.success(messageText);
    return true;
  } catch (err) {
    message.error("Failed to copy to clipboard");
    console.error("Failed to copy: ", err);
    return false;
  }
};