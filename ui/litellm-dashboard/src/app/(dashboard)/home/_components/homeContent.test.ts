import { describe, expect, it } from "vitest";
import type { BlogPost } from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";
import { DAY_ZERO_TAG, cardProvider, cardSummary, cardTitle, firstSentence, latestLaunches } from "./homeContent";

const post = (title: string, date: string, tags: string[] = [DAY_ZERO_TAG]): BlogPost => ({
  title,
  description: "d",
  date,
  url: `https://docs.litellm.ai/blog/${title}`,
  tags,
});

describe("latestLaunches", () => {
  it("keeps only posts tagged day 0 support", () => {
    const posts = [
      post("townhall", "2026-10-12", ["announcement"]),
      post("argon", "2026-10-13"),
      post("untagged", "2026-10-11", []),
    ];
    expect(latestLaunches(posts).map((p) => p.title)).toEqual(["argon"]);
  });

  it("orders launches by date, newest first, without mutating the input", () => {
    const posts = [post("old", "2026-09-01"), post("new", "2026-10-11"), post("mid", "2026-10-01")];
    expect(latestLaunches(posts).map((p) => p.title)).toEqual(["new", "mid", "old"]);
    expect(posts.map((p) => p.title)).toEqual(["old", "new", "mid"]);
  });

  it("keeps only the three newest so the cards fill one row", () => {
    const posts = [
      post("oldest", "2026-08-01"),
      post("a", "2026-10-03"),
      post("b", "2026-10-02"),
      post("c", "2026-10-01"),
    ];
    expect(latestLaunches(posts).map((p) => p.title)).toEqual(["a", "b", "c"]);
  });
});

describe("card fields", () => {
  const argon = post("Gemini 4 Argon API Pricing and Day 0 Support on LiteLLM", "2026-10-13", [
    "gemini",
    "gemini 4",
    "card_title:Gemini 4 Argon",
    "card_summary:Gemini 4 Argon costs $2.00 per 1M input tokens.",
    DAY_ZERO_TAG,
  ]);
  const legacy = {
    ...post("Day 0 Support: Grok 4.7", "2026-09-21", ["xai", DAY_ZERO_TAG]),
    description: "Day 0 support for Grok 4.7 on LiteLLM, at the same price as Grok 4.6. Use it anywhere.",
  };

  it("reads card_title and card_summary from the tags", () => {
    expect(cardTitle(argon)).toBe("Gemini 4 Argon");
    expect(cardSummary(argon)).toBe("Gemini 4 Argon costs $2.00 per 1M input tokens.");
  });

  it("falls back to the full title and the first sentence of the description", () => {
    expect(cardTitle(legacy)).toBe("Day 0 Support: Grok 4.7");
    expect(cardSummary(legacy)).toBe("Day 0 support for Grok 4.7 on LiteLLM, at the same price as Grok 4.6.");
    expect(firstSentence("No terminal punctuation here")).toBe("No terminal punctuation here");
    expect(firstSentence("Costs $0.10 per 1M tokens. Twice.")).toBe("Costs $0.10 per 1M tokens.");
  });

  it("treats an empty card tag as absent", () => {
    const blank = post("Full title", "2026-10-01", ["card_title: ", DAY_ZERO_TAG]);
    expect(cardTitle(blank)).toBe("Full title");
  });

  it("picks the provider logo from the first tag that is a litellm provider", () => {
    expect(cardProvider(argon)).toBe("gemini");
    expect(cardProvider(legacy)).toBe("xai");
    expect(cardProvider(post("x", "2026-10-01", ["nano banana", DAY_ZERO_TAG]))).toBeUndefined();
  });
});
