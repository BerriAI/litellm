import { z } from "zod";

export const SKILL_AGENTS = [
  { value: "codex", label: "Codex" },
  { value: "claude-code", label: "Claude Code" },
  { value: "cursor", label: "Cursor" },
  { value: "github-copilot", label: "GitHub Copilot" },
  { value: "gemini-cli", label: "Gemini CLI" },
] as const;

export type SkillAgent = (typeof SKILL_AGENTS)[number]["value"];
const skillsPackage = "skills@1.7.6";
const publicIndexSchema = z.object({
  skills: z.array(z.object({ name: z.string(), type: z.literal("archive"), url: z.string() })),
});

const shellQuote = (value: string): string => `'${value.replaceAll("'", "'\"'\"'")}'`;

const parseUrl = (value: string): URL | null => {
  try {
    const url = new URL(value);
    return /^https?:$/.test(url.protocol) && !url.username && !url.password ? url : null;
  } catch {
    return null;
  }
};

export const findPublicSkill = (index: unknown, archiveUrl: string) => {
  const parsed = publicIndexSchema.safeParse(index);
  return parsed.success
    ? parsed.data.skills.find((skill) => {
        const url = parseUrl(skill.url);
        if (!url || url.hash) return false;
        return (
          `${url.origin}${url.pathname}` === archiveUrl && [...url.searchParams.keys()].every((key) => key === "digest")
        );
      })
    : undefined;
};

interface SkillInstallOptions {
  gatewayUrl: string;
  archiveUrl: string;
  name: string;
  agent: SkillAgent;
  global: boolean;
  publicSkill?: { name: string };
  authHeaderName: string;
}

export function buildSkillInstallCommand(options: SkillInstallOptions): string | null {
  const gateway = parseUrl(options.gatewayUrl);
  const archive = parseUrl(options.archiveUrl);
  if (!gateway || gateway.search || gateway.hash) return null;
  if (!archive || archive.origin !== gateway.origin) return null;
  if (!/^[a-z0-9-]+$/.test(options.name)) return null;
  if (!/^[!#$%&'*+.^_`|~\da-z-]+$/i.test(options.authHeaderName)) return null;
  const gatewayUrl = gateway.href.replace(/\/+$/, "");
  if (archive.href !== `${gatewayUrl}/skill-hub/skills/${options.name}/archive`) return null;
  const scope = options.global ? " -g" : "";
  if (options.publicSkill) {
    return `DISABLE_TELEMETRY=1 npx ${skillsPackage} add ${shellQuote(gatewayUrl)} --skill ${shellQuote(options.publicSkill.name)} -a ${options.agent}${scope}`;
  }
  const loopback =
    /^(?:localhost|\[::1\]|127(?:\.\d{1,3}){3})$/.test(gateway.hostname) || gateway.hostname.endsWith(".localhost");
  if (gateway.protocol !== "https:" && !loopback) return null;
  return `bash <<'LITELLM_SKILL'
set -euo pipefail
set +x
skill_tmp=$(mktemp -d)
trap 'rm -rf "$skill_tmp"' EXIT
IFS= read -r -s -p 'LiteLLM API key: ' skill_key </dev/tty
export -n skill_key
printf '\\n'
printf '%s: Bearer %s\\n' ${shellQuote(options.authHeaderName)} "$skill_key" |
  curl -q --fail --silent --show-error --max-time 60 --header @- \\
    ${shellQuote(archive.href)} --output "$skill_tmp/skill.zip"
unset skill_key
unzip -q "$skill_tmp/skill.zip" -d "$skill_tmp/skill"
DISABLE_TELEMETRY=1 npx --yes ${skillsPackage} add "$skill_tmp/skill" --skill ${shellQuote(options.name)} -a ${options.agent} --copy --yes${scope}
LITELLM_SKILL`;
}
