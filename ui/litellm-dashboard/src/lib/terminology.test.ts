import { describe, expect, it } from "vitest";
import { toSiTerminology } from "./terminology";

describe("toSiTerminology", () => {
  it.each([
    ["AI Gateway", "SI Gateway"],
    ["AI GATEWAY", "SI GATEWAY"],
    ["Enable your AI assistant", "Enable your SI assistant"],
    ["AI-Discovered Competitors", "SI-Discovered Competitors"],
    ["protecting AI/ML workloads", "protecting SI/ML workloads"],
    ["the AI's response", "the SI's response"],
    ["Artificial Intelligence", "Super Intelligence"],
    ["artificial intelligence safety", "super intelligence safety"],
    ["Artificial intelligence", "Super intelligence"],
  ])("rewrites %j to %j", (input, expected) => {
    expect(toSiTerminology(input)).toBe(expected);
  });

  it.each(["OpenAI", "xAI's assistant", "AIM Security guardrails", "MAIN", "Chat"])(
    "leaves %j alone because AI is not a standalone word there",
    (input) => {
      expect(toSiTerminology(input)).toBe(input);
    },
  );

  it.each([
    [
      "Explain the EU AI Act's prohibition on AI criminal risk prediction.",
      "Explain the EU AI Act's prohibition on SI criminal risk prediction.",
    ],
    [
      "Article 5 prohibited AI practices under the European Union AI Act.",
      "Article 5 prohibited SI practices under the European Union AI Act.",
    ],
    [
      "Palo Alto Networks Prisma AI Runtime Security for securing AI applications",
      "Palo Alto Networks Prisma AI Runtime Security for securing SI applications",
    ],
    ["Cisco AI Defense Inspection API for AI apps", "Cisco AI Defense Inspection API for SI apps"],
    ["DeepKeep AI Firewall for AI security", "DeepKeep AI Firewall for SI security"],
    ["Defend AI Agentic Guardrails", "Defend AI Agentic Guardrails"],
    ["CyCraft XecGuard AI security gateway for AI apps", "CyCraft XecGuard AI security gateway for SI apps"],
    ["Airia AI security platform for responsible AI", "Airia AI security platform for responsible SI"],
  ])("keeps third-party names inside %j", (input, expected) => {
    expect(toSiTerminology(input)).toBe(expected);
  });
});
