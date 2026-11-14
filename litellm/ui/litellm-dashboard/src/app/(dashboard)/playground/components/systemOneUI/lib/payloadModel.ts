import { z } from "zod";
import { parseJson } from "./validatePayload";

const jsonObject = z.record(z.string(), z.unknown());

const parseObject = (raw: string): Record<string, unknown> | undefined => {
  const json = parseJson(raw);
  if (!json.ok) {
    return undefined;
  }
  const parsed = jsonObject.safeParse(json.value);
  return parsed.success ? parsed.data : undefined;
};

export const payloadModel = (raw: string): string | undefined => {
  const model = parseObject(raw)?.model;
  return typeof model === "string" && model !== "" ? model : undefined;
};

export const withPayloadModel = (raw: string, model: string): string | undefined => {
  const payload = parseObject(raw);
  if (payload === undefined) {
    return undefined;
  }
  const rest = Object.fromEntries(Object.entries(payload).filter(([key]) => key !== "model"));
  return JSON.stringify({ model, ...rest }, null, 2);
};
