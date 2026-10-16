import { describe, expect, it } from "vitest";

import type { RoutingGroup } from "./types";
import {
  argsForStrategy,
  buildRoutingGroupPayload,
  prioritiesForModels,
  toRoutingGroupFormValues,
  type RoutingGroupFormValues,
} from "./routingGroupPayload";

const values = (overrides: Partial<RoutingGroupFormValues> = {}): RoutingGroupFormValues => ({
  group_name: "fast-chat",
  models: ["gpt-4o"],
  routing_strategy: "simple-shuffle",
  routing_strategy_args: "",
  model_priorities: [{ model: "gpt-4o", priority: "1" }],
  ...overrides,
});

describe("buildRoutingGroupPayload", () => {
  it("sends a null args key for a strategy that takes no arguments", () => {
    expect(buildRoutingGroupPayload(values())).toStrictEqual({
      ok: true,
      group: {
        group_name: "fast-chat",
        models: ["gpt-4o"],
        routing_strategy: "simple-shuffle",
        routing_strategy_args: null,
      },
    });
  });

  it("parses the arguments for latency based routing", () => {
    const result = buildRoutingGroupPayload(
      values({ routing_strategy: "latency-based-routing", routing_strategy_args: '{"ttl": 3600}' }),
    );

    expect(result).toStrictEqual({
      ok: true,
      group: {
        group_name: "fast-chat",
        models: ["gpt-4o"],
        routing_strategy: "latency-based-routing",
        routing_strategy_args: { ttl: 3600 },
      },
    });
  });

  it("parses the arguments for usage based routing", () => {
    const result = buildRoutingGroupPayload(
      values({ routing_strategy: "usage-based-routing", routing_strategy_args: '{"ttl": 60}' }),
    );

    expect(result.ok && result.group.routing_strategy_args).toStrictEqual({ ttl: 60 });
  });

  it("drops arguments belonging to a strategy that does not take them", () => {
    const result = buildRoutingGroupPayload(
      values({ routing_strategy: "least-busy", routing_strategy_args: '{"ttl": 3600}' }),
    );

    expect(result.ok && result.group.routing_strategy_args).toBeNull();
  });

  it("treats whitespace-only arguments as absent", () => {
    const result = buildRoutingGroupPayload(
      values({ routing_strategy: "latency-based-routing", routing_strategy_args: "   \n  " }),
    );

    expect(result.ok && result.group.routing_strategy_args).toBeNull();
  });

  it("reports invalid JSON instead of a payload", () => {
    expect(
      buildRoutingGroupPayload(values({ routing_strategy: "latency-based-routing", routing_strategy_args: "{ttl:}" })),
    ).toStrictEqual({ ok: false, field: "routing_strategy_args", message: "Must be valid JSON" });
  });

  it("trims the group name", () => {
    const result = buildRoutingGroupPayload(values({ group_name: "  fast-chat  " }));

    expect(result.ok && result.group.group_name).toBe("fast-chat");
  });

  it("passes the selected models through untouched", () => {
    const models = ["gpt-4o", "claude-sonnet", "gemini-pro"];
    const result = buildRoutingGroupPayload(values({ models }));

    expect(result.ok && result.group.models).toStrictEqual(models);
  });
});

describe("priority payloads", () => {
  const priorityValues = (overrides: Partial<RoutingGroupFormValues> = {}): RoutingGroupFormValues => {
    const defaults = {
      models: ["preferred", "backup", "peer"],
      routing_strategy: "priority",
      model_priorities: [
        { model: "preferred", priority: "1" },
        { model: "backup", priority: "2" },
        { model: "peer", priority: "1" },
      ],
      ...overrides,
    };
    return values(defaults);
  };

  it("sends explicit membership priorities, including ties, without strategy arguments", () => {
    expect(buildRoutingGroupPayload(priorityValues())).toStrictEqual({
      ok: true,
      group: {
        group_name: "fast-chat",
        models: ["preferred", "backup", "peer"],
        routing_strategy: "priority",
        routing_strategy_args: null,
        model_priorities: { preferred: 1, backup: 2, peer: 1 },
      },
    });
  });

  it.each(["", " ", "0", "-1", "1.5", "NaN", "Infinity", "9007199254740992", "1.000000000000000001", "0x10"])(
    "rejects the invalid priority %j without silently defaulting",
    (priority) => {
      expect(
        buildRoutingGroupPayload(
          priorityValues({
            model_priorities: [
              { model: "preferred", priority },
              { model: "backup", priority: "2" },
              { model: "peer", priority: "1" },
            ],
          }),
        ),
      ).toStrictEqual({
        ok: false,
        field: "model_priorities",
        message: "Priorities must be whole numbers from 1 to 9007199254740991",
      });
    },
  );

  it.each([
    [],
    [{ model: "preferred", priority: "1" }],
    [
      { model: "preferred", priority: "1" },
      { model: "backup", priority: "2" },
      { model: "unknown", priority: "3" },
    ],
    [
      { model: "preferred", priority: "1" },
      { model: "preferred", priority: "2" },
      { model: "peer", priority: "1" },
    ],
  ])("requires exactly the selected membership priorities", (...model_priorities) => {
    expect(buildRoutingGroupPayload(priorityValues({ model_priorities })).ok).toBe(false);
  });

  it("omits membership priorities when another strategy is selected", () => {
    const result = buildRoutingGroupPayload(priorityValues({ routing_strategy: "simple-shuffle" }));
    expect(result.ok && result.group).not.toHaveProperty("model_priorities");
  });

  it("preserves model names with dots and object prototype names as own payload keys", () => {
    const models = ["provider/model.v1", "__proto__", "constructor"];
    const result = buildRoutingGroupPayload(
      priorityValues({ models, model_priorities: prioritiesForModels(models, []) }),
    );
    expect(result.ok && Object.entries(result.group.model_priorities ?? {})).toStrictEqual([
      ["provider/model.v1", 1],
      ["__proto__", 2],
      ["constructor", 3],
    ]);
  });

  it("keeps persisted values, missing values and unused entries visible until explicitly repaired", () => {
    const stored: RoutingGroup = {
      group_name: "priority-group",
      models: ["preferred", "backup", "peer"],
      routing_strategy: "priority",
      model_priorities: { unused: 8, peer: -1, preferred: 7 },
    };
    const form = toRoutingGroupFormValues(stored, []);
    expect(form.model_priorities).toStrictEqual([
      { model: "preferred", priority: "7" },
      { model: "backup", priority: "" },
      { model: "peer", priority: "-1" },
      { model: "unused", priority: "8" },
    ]);
    expect(buildRoutingGroupPayload(form).ok).toBe(false);
  });

  it("round-trips explicit priorities independently of persisted map order", () => {
    const stored: RoutingGroup = {
      group_name: "priority-group",
      models: ["preferred", "backup"],
      routing_strategy: "priority",
      routing_strategy_args: null,
      model_priorities: { backup: Number.MAX_SAFE_INTEGER, preferred: 4 },
    };
    expect(buildRoutingGroupPayload(toRoutingGroupFormValues(stored, []))).toStrictEqual({ ok: true, group: stored });
  });

  it("retains edited priorities and defaults new members when membership is explicitly changed", () => {
    expect(
      prioritiesForModels(
        ["preferred", "new"],
        [
          { model: "preferred", priority: "5" },
          { model: "removed", priority: "2" },
        ],
      ),
    ).toStrictEqual([
      { model: "preferred", priority: "5" },
      { model: "new", priority: "2" },
    ]);
  });
});

describe("argsForStrategy", () => {
  it("keeps the arguments when the new strategy still takes them", () => {
    expect(argsForStrategy("usage-based-routing", '{"ttl": 60}')).toBe('{"ttl": 60}');
  });

  it("clears the arguments when the new strategy takes none", () => {
    expect(argsForStrategy("simple-shuffle", '{"ttl": 60}')).toBe("");
  });
});

describe("toRoutingGroupFormValues", () => {
  it("falls back to empty values and the first available strategy when creating", () => {
    const expected: RoutingGroupFormValues = {
      group_name: "",
      models: [],
      routing_strategy: "least-busy",
      routing_strategy_args: "",
      model_priorities: [],
    };

    expect(toRoutingGroupFormValues(null, ["least-busy", "simple-shuffle"])).toStrictEqual(expected);
  });

  it("falls back to simple-shuffle when no strategy is available", () => {
    expect(toRoutingGroupFormValues(null, []).routing_strategy).toBe("simple-shuffle");
  });

  it("pretty-prints the stored arguments", () => {
    const stored: RoutingGroup = {
      group_name: "latency-group",
      models: ["gpt-4o"],
      routing_strategy: "latency-based-routing",
      routing_strategy_args: { ttl: 3600 },
    };
    const expected: RoutingGroupFormValues = {
      group_name: "latency-group",
      models: ["gpt-4o"],
      routing_strategy: "latency-based-routing",
      routing_strategy_args: '{\n  "ttl": 3600\n}',
      model_priorities: [{ model: "gpt-4o", priority: "1" }],
    };

    expect(toRoutingGroupFormValues(stored, [])).toStrictEqual(expected);
  });

  it("leaves the arguments blank when the stored group has none", () => {
    const stored: RoutingGroup = {
      group_name: "g",
      models: [],
      routing_strategy: "simple-shuffle",
      routing_strategy_args: null,
    };

    expect(toRoutingGroupFormValues(stored, []).routing_strategy_args).toBe("");
  });

  it("carries only bound fields, never the rest of the record", () => {
    expect(
      Object.keys(
        toRoutingGroupFormValues({ group_name: "g", models: [], routing_strategy: "simple-shuffle" }, []),
      ).sort(),
    ).toStrictEqual(["group_name", "model_priorities", "models", "routing_strategy", "routing_strategy_args"]);
  });
});
