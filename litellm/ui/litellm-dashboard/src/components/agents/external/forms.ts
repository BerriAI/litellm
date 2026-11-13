import { createParser } from "nuqs";

export const pageParser = createParser({
  parse: (value) => (/^[1-9]\d*$/.test(value) && Number.isSafeInteger(Number(value)) ? Number(value) : null),
  serialize: String,
}).withDefault(1);

export const displayTime = (value?: string | null) => (value ? new Date(value).toLocaleString() : "No events yet");
