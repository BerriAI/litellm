import { describe, expect, it } from "vitest";
import type { BlogPost } from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";
import { DAY_ZERO_TAG, latestLaunches } from "./homeContent";

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
