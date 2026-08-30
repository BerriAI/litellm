import { z } from "zod/v4";

const IPV4 = "(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]\\d|\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]\\d|\\d)){3}";
const SEG = "[a-fA-F\\d]{1,4}";
const IPV6 =
  `(?:(?:${SEG}:){7}(?:${SEG}|:)|` +
  `(?:${SEG}:){6}(?:${IPV4}|:${SEG}|:)|` +
  `(?:${SEG}:){5}(?::${IPV4}|(?::${SEG}){1,2}|:)|` +
  `(?:${SEG}:){4}(?:(?::${SEG}){0,1}:${IPV4}|(?::${SEG}){1,3}|:)|` +
  `(?:${SEG}:){3}(?:(?::${SEG}){0,2}:${IPV4}|(?::${SEG}){1,4}|:)|` +
  `(?:${SEG}:){2}(?:(?::${SEG}){0,3}:${IPV4}|(?::${SEG}){1,5}|:)|` +
  `(?:${SEG}:){1}(?:(?::${SEG}){0,4}:${IPV4}|(?::${SEG}){1,6}|:)|` +
  `(?::(?:(?::${SEG}){0,5}:${IPV4}|(?::${SEG}){1,7}|:)))(?:%[0-9a-zA-Z]{1,})?`;
const HOST = "(?:(?:[a-z\\u00a1-\\uffff0-9][-_]*)*[a-z\\u00a1-\\uffff0-9]+)";
const DOMAIN = "(?:\\.(?:[a-z\\u00a1-\\uffff0-9]-*)*[a-z\\u00a1-\\uffff0-9]+)*";
const TLD = "(?:\\.(?:[a-z\\u00a1-\\uffff]{2,}))";

const URL_RULE_PATTERN = new RegExp(
  `(?:^(?:(?:(?:[a-z]+:)?//)|www\\.)(?:\\S+(?::\\S*)?@)?` +
    `(?:localhost|${IPV4}|${IPV6}|${HOST}${DOMAIN}${TLD})` +
    `(?::\\d{2,5})?(?:[/?#][^\\s"]*)?$)`,
  "i",
);

const isUrl = (value: string): boolean => value.length <= 2048 && URL_RULE_PATTERN.test(value);

const pluginShape = {
  name: z.string().min(1, "Required"),
  display_name: z.string().min(1, "Required"),
  url: z
    .string()
    .min(1, "Required")
    .refine((value) => value === "" || isUrl(value), "Must be a valid URL"),
  plugin_key: z.string().optional(),
};

export const pluginSchema = z.object(pluginShape);

export type PluginFormValues = z.output<typeof pluginSchema>;
