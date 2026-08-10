import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const STAGE_ONE_FILES = [
  "src/components/DashboardHeader.tsx",
  "src/components/Navbar/ViewSwitcher.tsx",
  "src/components/navbar.tsx",
  "src/components/LanguageSelector/LanguageSelector.tsx",
  "src/app/login/LoginPage.tsx",
  "src/app/onboarding/OnboardingForm.tsx",
  "src/app/onboarding/OnboardingFormBody.tsx",
  "src/app/onboarding/OnboardingErrorView.tsx",
  "src/app/onboarding/page.tsx",
  "src/components/public_model_hub.tsx",
  "src/app/model_hub/page.tsx",
  "src/app/model_hub_table/page.tsx",
  "src/app/mcp/oauth/callback/page.tsx",
];

// These values are product names, identifiers, or protocol terms. Translating
// them would alter commands, configuration, URLs, or externally defined names.
const TECHNICAL_LITERAL_ALLOWLIST = [
  { pattern: /^(?:🚅 )?LiteLLM(?: Brand)?$/, reason: "product name" },
  {
    pattern: /^(?:MASTER_KEY|PROXY_ADMIN_ID|DISABLE_ADMIN_UI=False|UI_USERNAME|AUTO_REDIRECT_UI_LOGIN_TO_SSO=true)$/,
    reason: "configuration identifier",
  },
  { pattern: /^(?:SSO|URL|MCP|A2A|OpenID|OAuth)$/, reason: "protocol or industry term" },
  { pattern: /^(?:admin|Admin)$/, reason: "fixed account role" },
  { pattern: /^OR$/, reason: "compact authentication separator" },
  { pattern: /^(?:: )?v$/, reason: "version prefix" },
];

function normalizedText(value) {
  return value.replace(/\s+/g, " ").trim();
}

function isTechnicalLiteral(text) {
  return TECHNICAL_LITERAL_ALLOWLIST.some(({ pattern }) => pattern.test(text));
}

function tagName(node) {
  if (ts.isJsxElement(node)) return node.openingElement.tagName.getText();
  return undefined;
}

function isInsideTechnicalMarkup(node) {
  for (let parent = node.parent; parent; parent = parent.parent) {
    if (["code", "pre"].includes(tagName(parent)?.toLowerCase())) return true;
  }
  return false;
}

export function auditSource(source, file = "source.tsx") {
  const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const findings = [];
  const seen = new Set();

  const record = (node, rawText, kind) => {
    const text = normalizedText(rawText);
    if (!/[A-Za-z]/.test(text) || isTechnicalLiteral(text)) return;

    const start = node.getStart(sourceFile);
    const key = `${start}:${kind}:${text}`;
    if (seen.has(key)) return;
    seen.add(key);
    findings.push({ file, line: sourceFile.getLineAndCharacterOfPosition(start).line + 1, text, kind });
  };

  const visit = (node) => {
    if (ts.isJsxText(node) && !isInsideTechnicalMarkup(node)) {
      record(node, node.getText(sourceFile), "jsx-text");
    } else if (ts.isJsxAttribute(node) && node.initializer && ts.isStringLiteral(node.initializer)) {
      const name = node.name.getText(sourceFile);
      if (["placeholder", "title", "aria-label", "alt"].includes(name)) {
        record(node, node.initializer.text, `jsx-${name}`);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);

  return findings;
}

export function auditFiles(files = STAGE_ONE_FILES) {
  return files.flatMap((file) => auditSource(readFileSync(file, "utf8"), file));
}

function runCli() {
  const findings = auditFiles();
  if (findings.length === 0) {
    console.log(`Localization audit passed (${STAGE_ONE_FILES.length} shared UI files).`);
    return;
  }

  console.error("Raw user-facing English found in localized shared UI:");
  for (const finding of findings) {
    console.error(`${finding.file}:${finding.line} [${finding.kind}] ${finding.text}`);
  }
  process.exitCode = 1;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  runCli();
}
