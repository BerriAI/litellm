import { isExternalAssetSrc } from "@/lib/assetPaths";

export type LogoTreatment = "invert" | "plate";

const BUNDLED_LOGO_PATH = /(?:\/assets\/logos\/|\/_next\/static\/media\/)/;

const TREATMENT_BY_ASSET: Readonly<Record<string, LogoTreatment>> = {
  "baseten.svg": "invert",
  "cursor.svg": "invert",
  "enkrypt_ai.avif": "invert",
  "friendli.svg": "invert",
  "github.svg": "invert",
  "github_copilot.svg": "invert",
  "lago.svg": "invert",
  "lambda.svg": "invert",
  "langflow.svg": "invert",
  "lmstudio.svg": "invert",
  "moonshot.svg": "invert",
  "nebius.svg": "invert",
  "notion.svg": "invert",
  "ollama.svg": "invert",
  "openrouter.svg": "invert",
  "promptguard.svg": "invert",
  "recraft.svg": "invert",
  "replicate.svg": "invert",
  "runway.png": "invert",
  "scx_ai.svg": "invert",
  "secret_detect.png": "invert",
  "topaz.svg": "invert",
  "v0.svg": "invert",
  "vercel.svg": "invert",
  "watsonx.svg": "invert",
  "aiml_api.svg": "plate",
  "akto.svg": "plate",
  "aws.svg": "plate",
  "deepkeep.svg": "plate",
  "fireworks.svg": "plate",
  "llm_guard.png": "plate",
  "pangea.png": "plate",
  "repelloai.png": "plate",
  "sambanova.svg": "plate",
  "sentry.svg": "plate",
  "valkey.svg": "plate",
};

const basenameOf = (src: string): string | undefined => src.split(/[?#]/)[0].split("/").pop() || undefined;

const withoutBundlerHash = (basename: string): string | undefined => {
  const parts = basename.split(".");
  return parts.length < 2 ? undefined : `${parts[0]}.${parts[parts.length - 1]}`;
};

export const logoTreatmentFor = (src: string | null | undefined): LogoTreatment | undefined => {
  if (!src || isExternalAssetSrc(src) || !BUNDLED_LOGO_PATH.test(src)) return undefined;
  const basename = basenameOf(src);
  const key = basename === undefined ? undefined : withoutBundlerHash(basename);
  return key === undefined ? undefined : TREATMENT_BY_ASSET[key];
};
