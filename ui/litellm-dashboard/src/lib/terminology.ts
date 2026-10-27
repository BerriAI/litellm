export type Term = (text: string) => string;

export const keepTerminology: Term = (text) => text;

const THIRD_PARTY_NAMES: readonly string[] = [
  "EU AI Act",
  "European Union AI Act",
  "Cisco AI Defense",
  "DeepKeep AI Firewall",
  "Defend AI",
  "Prisma AI Runtime Security",
  "XecGuard AI",
  "Airia AI",
];

const THIRD_PARTY_NAME_PATTERN = new RegExp(`(${THIRD_PARTY_NAMES.join("|")})`);

const swapTerms = (text: string): string =>
  text
    .replace(
      /\b([Aa])rtificial ([Ii])ntelligence\b/g,
      (_match: string, a: string, i: string) => `${a === "A" ? "S" : "s"}uper ${i}ntelligence`,
    )
    .replace(/\bAI\b/g, "SI");

export const toSiTerminology: Term = (text) =>
  text
    .split(THIRD_PARTY_NAME_PATTERN)
    .map((segment, index) => (index % 2 === 1 ? segment : swapTerms(segment)))
    .join("");
