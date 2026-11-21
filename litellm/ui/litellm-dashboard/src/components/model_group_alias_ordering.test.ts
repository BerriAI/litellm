import { describe, expect, it } from "vitest";
import {
  DEFAULT_ALIAS_DISPLAY,
  movePinnedAlias,
  orderAliases,
  type AliasDisplayPreferences,
  type AliasItem,
} from "./model_group_alias_ordering";

const row = (aliasName: string, targetModelGroup: string): AliasItem => ({
  id: aliasName,
  aliasName,
  targetModelGroup,
});
const names = (aliases: readonly AliasItem[]): string[] => aliases.map((alias) => alias.aliasName);

describe("orderAliases", () => {
  const aliases = Object.freeze([row("Zebra", "alpha10"), row("alpha10", "beta"), row("alpha2", "Alpha2")]);

  it.each<{ sortBy: AliasDisplayPreferences["sortBy"]; descending: boolean; expected: string[] }>([
    { sortBy: "aliasName", descending: false, expected: ["alpha2", "alpha10", "Zebra"] },
    { sortBy: "aliasName", descending: true, expected: ["Zebra", "alpha10", "alpha2"] },
    { sortBy: "targetModelGroup", descending: false, expected: ["alpha2", "Zebra", "alpha10"] },
    { sortBy: "targetModelGroup", descending: true, expected: ["alpha10", "Zebra", "alpha2"] },
  ])("sorts $sortBy with descending=$descending without changing the input", ({ sortBy, descending, expected }) => {
    expect(names(orderAliases(aliases, { ...DEFAULT_ALIAS_DISPLAY, sortBy, descending }))).toEqual(expected);
    expect(names(aliases)).toEqual(["Zebra", "alpha10", "alpha2"]);
  });

  it.each([false, true])(
    "breaks equal-target ties by alias name regardless of input order, descending=%s",
    (descending) => {
      const tied = [row("beta", "shared"), row("Alpha", "SHARED"), row("gamma", "shared")];
      const preferences: AliasDisplayPreferences = { ...DEFAULT_ALIAS_DISPLAY, sortBy: "targetModelGroup", descending };
      expect(names(orderAliases(tied, preferences))).toEqual(["Alpha", "beta", "gamma"]);
      expect(names(orderAliases([...tied].reverse(), preferences))).toEqual(["Alpha", "beta", "gamma"]);
    },
  );

  it.each([
    { descending: false, remaining: ["alpha", "beta", "subscription/a", "subscription/c"] },
    { descending: true, remaining: ["beta", "alpha", "subscription/c", "subscription/a"] },
  ])("keeps ordered pins first and alias-prefix matches last, descending=$descending", ({ descending, remaining }) => {
    const grouped = [
      row("subscription/c", "c"),
      row("beta", "b"),
      row("ordinary", "y"),
      row("subscription/a", "a"),
      row("alpha", "subscription/x"),
      row("subscription/z", "z"),
    ];
    const preferences = {
      ...DEFAULT_ALIAS_DISPLAY,
      descending,
      pinned: ["missing", "subscription/z", "ordinary"],
      lastPrefix: "subscription/",
    };
    expect(names(orderAliases(grouped, preferences))).toEqual(["subscription/z", "ordinary", ...remaining]);
  });
});

describe("movePinnedAlias", () => {
  const pinned = Object.freeze(["first", "middle", "last"]);

  it.each<{ name: string; direction: -1 | 1; expected: string[] }>([
    { name: "middle", direction: -1, expected: ["middle", "first", "last"] },
    { name: "middle", direction: 1, expected: ["first", "last", "middle"] },
    { name: "first", direction: -1, expected: ["first", "middle", "last"] },
    { name: "last", direction: 1, expected: ["first", "middle", "last"] },
    { name: "missing", direction: -1, expected: ["first", "middle", "last"] },
    { name: "missing", direction: 1, expected: ["first", "middle", "last"] },
  ])("moves $name in direction $direction without changing the input", ({ name, direction, expected }) => {
    expect(movePinnedAlias(pinned, name, direction)).toEqual(expected);
    expect(pinned).toEqual(["first", "middle", "last"]);
  });
});
