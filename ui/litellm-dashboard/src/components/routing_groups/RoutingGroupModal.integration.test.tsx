import userEvent from "@testing-library/user-event";
import { fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen } from "@/../tests/test-utils";

import RoutingGroupModal from "./RoutingGroupModal";
import type { RoutingGroup } from "./types";

const STRATEGIES = ["simple-shuffle", "latency-based-routing", "usage-based-routing", "priority"];
const MODEL_OPTIONS = ["gpt-4o", "claude-sonnet", "gemini-pro"];
const STRATEGY_DESCRIPTIONS = { "simple-shuffle": "Spreads requests evenly across the group." };

const EXPECTED_STORED_PAYLOAD: RoutingGroup = {
  group_name: "already-taken",
  models: ["gpt-4o", "claude-sonnet"],
  routing_strategy: "latency-based-routing",
  routing_strategy_args: { ttl: 3600 },
};

const SEEDED_CREATE: RoutingGroup = { group_name: "", models: ["gemini-pro"], routing_strategy: "simple-shuffle" };

const EXPECTED_SLASH_AND_SPACE_PAYLOAD: RoutingGroup = {
  group_name: "team a/fast chat",
  models: ["gemini-pro"],
  routing_strategy: "simple-shuffle",
  routing_strategy_args: null,
};

const STORED_GROUP: RoutingGroup = {
  group_name: "already-taken",
  models: ["gpt-4o", "claude-sonnet"],
  routing_strategy: "latency-based-routing",
  routing_strategy_args: { ttl: 3600 },
};

const STORED_GROUP_NULL_ARGS: RoutingGroup = {
  group_name: "already-taken",
  models: ["gpt-4o"],
  routing_strategy: "latency-based-routing",
  routing_strategy_args: null,
};

const STORED_PRIORITY_GROUP: RoutingGroup = {
  group_name: "preferred-chat",
  models: ["gpt-4o", "claude-sonnet"],
  routing_strategy: "priority",
  routing_strategy_args: null,
  model_priorities: { "gpt-4o": 3, "claude-sonnet": 7 },
};

const EXPECTED_NULL_ARGS_PAYLOAD: RoutingGroup = {
  group_name: "already-taken",
  models: ["gpt-4o"],
  routing_strategy: "latency-based-routing",
  routing_strategy_args: null,
};

const renderModal = (overrides: Partial<React.ComponentProps<typeof RoutingGroupModal>> = {}) => {
  const onSubmit = vi.fn();
  const onClose = vi.fn();
  renderWithProviders(
    <RoutingGroupModal
      open
      mode="create"
      initialValue={null}
      availableStrategies={STRATEGIES}
      strategyDescriptions={STRATEGY_DESCRIPTIONS}
      modelOptions={MODEL_OPTIONS}
      existingGroupNames={["already-taken", "other-group"]}
      groupNameByModel={{}}
      onClose={onClose}
      onSubmit={onSubmit}
      {...overrides}
    />,
  );
  return { onSubmit, onClose };
};

const typeName = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  const input = screen.getByLabelText("Group Name");
  await user.clear(input);
  await user.type(input, name);
};

const setArgs = async (user: ReturnType<typeof userEvent.setup>, json: string) => {
  const textarea = screen.getByLabelText("Strategy Arguments (JSON)");
  await user.clear(textarea);
  if (json) {
    await user.type(textarea, json);
  }
};

const pickModels = async (user: ReturnType<typeof userEvent.setup>, ...models: string[]) => {
  await user.click(screen.getByLabelText("Models"));
  for (const model of models) {
    await user.click(await screen.findByRole("option", { name: model }));
  }
};

const pickStrategy = async (user: ReturnType<typeof userEvent.setup>, strategy: string) => {
  await user.click(screen.getByLabelText("Routing Strategy"));
  await user.click(await screen.findByRole("option", { name: strategy }));
};

const save = async (user: ReturnType<typeof userEvent.setup>, name: string) =>
  await user.click(screen.getByRole("button", { name }));

describe("RoutingGroupModal", () => {
  it("creates a priority group with editable defaults and members already used by another group", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({ groupNameByModel: { "gpt-4o": "legacy-group" } });

    fireEvent.change(screen.getByLabelText("Group Name"), { target: { value: "preferred-chat" } });
    await pickModels(user, "gpt-4o", "claude-sonnet");
    await pickStrategy(user, "Priority");

    expect(screen.getByLabelText("Priority for gpt-4o")).toHaveValue("1");
    expect(screen.getByLabelText("Priority for claude-sonnet")).toHaveValue("2");
    fireEvent.change(screen.getByLabelText("Priority for claude-sonnet"), { target: { value: "1" } });
    await save(user, "Create Group");

    const expected: RoutingGroup = {
      group_name: "preferred-chat",
      models: ["gpt-4o", "claude-sonnet"],
      routing_strategy: "priority",
      routing_strategy_args: null,
      model_priorities: { "gpt-4o": 1, "claude-sonnet": 1 },
    };
    expect(onSubmit).toHaveBeenCalledWith(expected);
  });

  it("preserves explicit priorities through edit for model names that are dictionary keys", async () => {
    const user = userEvent.setup();
    const stored: RoutingGroup = {
      group_name: "preferred-chat",
      models: ["provider/model.v1", "constructor", "__proto__"],
      routing_strategy: "priority",
      routing_strategy_args: null,
      model_priorities: Object.fromEntries([
        ["__proto__", 3],
        ["constructor", 8],
        ["provider/model.v1", 2],
      ]),
    };
    const { onSubmit } = renderModal({ mode: "edit", initialValue: stored, modelOptions: stored.models });

    expect(screen.getByLabelText("Priority for provider/model.v1")).toHaveValue("2");
    expect(screen.getByLabelText("Priority for constructor")).toHaveValue("8");
    await save(user, "Save Changes");

    expect(onSubmit).toHaveBeenCalledWith(stored);
  });

  it.each([
    { catalog: "omits Priority", strategies: ["simple-shuffle"] },
    { catalog: "includes Priority", strategies: STRATEGIES },
  ])("keeps unsaved priorities when switching away and back while the catalog $catalog", async ({ strategies }) => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      mode: "edit",
      initialValue: STORED_PRIORITY_GROUP,
      availableStrategies: strategies,
    });

    fireEvent.change(screen.getByLabelText("Priority for gpt-4o"), { target: { value: "9" } });
    await pickStrategy(user, "simple-shuffle");
    expect(screen.queryByLabelText("Priority for gpt-4o")).not.toBeInTheDocument();
    await pickStrategy(user, "Priority");
    expect(screen.getByLabelText("Priority for gpt-4o")).toHaveValue("9");
    expect(screen.getByLabelText("Priority for claude-sonnet")).toHaveValue("7");
    await save(user, "Save Changes");

    expect(onSubmit).toHaveBeenCalledWith({
      ...STORED_PRIORITY_GROUP,
      model_priorities: { "gpt-4o": 9, "claude-sonnet": 7 },
    });
  });

  it("offers only the backend catalog when creating a group", async () => {
    const user = userEvent.setup();
    renderModal({ initialValue: STORED_PRIORITY_GROUP, availableStrategies: ["simple-shuffle"] });

    await user.click(screen.getByLabelText("Routing Strategy"));

    expect(await screen.findByRole("option", { name: "simple-shuffle" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Priority" })).not.toBeInTheDocument();
  });

  it("shows invalid stored priorities and lets the user repair them before saving", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      mode: "edit",
      initialValue: {
        group_name: "preferred-chat",
        models: ["gpt-4o"],
        routing_strategy: "priority",
        model_priorities: { "gpt-4o": 0, unused: 9 },
      },
    });

    expect(screen.getByLabelText("Priority for gpt-4o")).toHaveValue("0");
    expect(screen.getByText("Model is not selected")).toBeInTheDocument();
    await save(user, "Save Changes");
    expect(onSubmit).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Remove priority for unused" }));
    await save(user, "Save Changes");
    expect(await screen.findByText("Priorities must be whole numbers from 1 to 9007199254740991")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Priority for gpt-4o"), { target: { value: "4" } });
    await save(user, "Save Changes");

    const expected: RoutingGroup = {
      group_name: "preferred-chat",
      models: ["gpt-4o"],
      routing_strategy: "priority",
      routing_strategy_args: null,
      model_priorities: { "gpt-4o": 4 },
    };
    expect(onSubmit).toHaveBeenCalledWith(expected);
  });

  it("restores the legacy ownership restriction when switching away from priority", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      mode: "edit",
      initialValue: {
        group_name: "preferred-chat",
        models: ["gpt-4o"],
        routing_strategy: "priority",
        model_priorities: { "gpt-4o": 1 },
      },
      groupNameByModel: { "gpt-4o": "legacy-group" },
    });

    await pickStrategy(user, "simple-shuffle");
    await save(user, "Save Changes");

    expect(await screen.findByText(/Already claimed: gpt-4o/)).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits an untouched edit of a group whose stored arguments are null", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({ mode: "edit", initialValue: STORED_GROUP_NULL_ARGS });

    await save(user, "Save Changes");

    expect(onSubmit).toHaveBeenCalledWith(EXPECTED_NULL_ARGS_PAYLOAD);
  });

  it("submits an untouched edit with the stored models, strategy and parsed arguments", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({ mode: "edit", initialValue: STORED_GROUP });

    await save(user, "Save Changes");

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toStrictEqual(EXPECTED_STORED_PAYLOAD);
  });

  it("carries a typed group name into the payload", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({ initialValue: SEEDED_CREATE });

    await typeName(user, "fast-chat");
    await save(user, "Create Group");

    const expected: RoutingGroup = {
      group_name: "fast-chat",
      models: ["gemini-pro"],
      routing_strategy: "simple-shuffle",
      routing_strategy_args: null,
    };
    expect(onSubmit.mock.calls[0][0]).toStrictEqual(expected);
  });

  it("sends null arguments when the selected strategy does not take them", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      mode: "edit",
      initialValue: { ...STORED_GROUP, routing_strategy: "simple-shuffle" },
    });

    expect(screen.queryByLabelText("Strategy Arguments (JSON)")).not.toBeInTheDocument();
    await save(user, "Save Changes");

    const expected: RoutingGroup = {
      group_name: "already-taken",
      models: ["gpt-4o", "claude-sonnet"],
      routing_strategy: "simple-shuffle",
      routing_strategy_args: null,
    };
    expect(onSubmit.mock.calls[0][0]).toStrictEqual(expected);
  });

  it("sends null arguments when the argument box is emptied", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({ mode: "edit", initialValue: STORED_GROUP });

    await setArgs(user, "");
    await save(user, "Save Changes");

    expect(onSubmit.mock.calls[0][0]?.routing_strategy_args).toBeNull();
  });

  it("edits the arguments into the payload", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({ mode: "edit", initialValue: STORED_GROUP });

    await setArgs(user, '{{"ttl": 60, "lowest_latency_buffer": 0}');
    await save(user, "Save Changes");

    expect(onSubmit.mock.calls[0][0]?.routing_strategy_args).toStrictEqual({ ttl: 60, lowest_latency_buffer: 0 });
  });

  it("blocks the save and flags the field when the arguments are not valid JSON", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({ mode: "edit", initialValue: STORED_GROUP });

    await setArgs(user, "not json");
    await save(user, "Save Changes");

    expect(await screen.findByText("Must be valid JSON")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("requires a group name", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      initialValue: { group_name: "", models: ["gemini-pro"], routing_strategy: "simple-shuffle" },
    });

    await save(user, "Create Group");

    expect(await screen.findByText("Group name is required")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("requires at least one model", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal();

    await typeName(user, "no-models");
    await save(user, "Create Group");

    expect(await screen.findByText("Select at least one model")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("rejects a name longer than 64 characters", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      initialValue: { group_name: "", models: ["gemini-pro"], routing_strategy: "simple-shuffle" },
    });

    await typeName(user, "a".repeat(65));
    await save(user, "Create Group");

    expect(await screen.findByText("Must be 64 characters or fewer")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("accepts a name with slashes and spaces, since the backend does", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      initialValue: { group_name: "", models: ["gemini-pro"], routing_strategy: "simple-shuffle" },
    });

    await typeName(user, "team a/fast chat");
    await save(user, "Create Group");

    expect(onSubmit).toHaveBeenCalledWith(EXPECTED_SLASH_AND_SPACE_PAYLOAD);
  });

  it("rejects a whitespace-only name as missing", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      initialValue: { group_name: "", models: ["gemini-pro"], routing_strategy: "simple-shuffle" },
    });

    await typeName(user, "   ");
    await save(user, "Create Group");

    expect(await screen.findByText("Group name is required")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("rejects a name another group already uses, ignoring case", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({
      initialValue: { group_name: "", models: ["gemini-pro"], routing_strategy: "simple-shuffle" },
    });

    await typeName(user, "Other-Group");
    await save(user, "Create Group");

    expect(await screen.findByText("A group with this name already exists")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("locks the name in edit mode and pretty-prints the stored arguments", () => {
    renderModal({ mode: "edit", initialValue: STORED_GROUP });

    expect(screen.getByLabelText("Group Name")).toHaveValue("already-taken");
    expect(screen.getByLabelText("Group Name")).toBeDisabled();
    expect(screen.getByLabelText("Strategy Arguments (JSON)")).toHaveValue('{\n  "ttl": 3600\n}');
  });

  it("carries picked models and a picked strategy into the payload", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal();

    await typeName(user, "probe-group");
    await pickModels(user, "gpt-4o", "claude-sonnet");
    await pickStrategy(user, "latency-based-routing");
    await setArgs(user, '{{"ttl": 99}');
    await save(user, "Create Group");

    const expected: RoutingGroup = {
      group_name: "probe-group",
      models: ["gpt-4o", "claude-sonnet"],
      routing_strategy: "latency-based-routing",
      routing_strategy_args: { ttl: 99 },
    };
    expect(onSubmit.mock.calls[0][0]).toStrictEqual(expected);
  });

  it("forgets arguments typed before the strategy stopped taking them", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal();

    await typeName(user, "probe-group");
    await pickModels(user, "gpt-4o");
    await pickStrategy(user, "latency-based-routing");
    await setArgs(user, '{{"ttl": 99}');
    await pickStrategy(user, "simple-shuffle");
    expect(screen.queryByLabelText("Strategy Arguments (JSON)")).not.toBeInTheDocument();
    await pickStrategy(user, "latency-based-routing");

    expect(screen.getByLabelText("Strategy Arguments (JSON)")).toHaveValue("");

    await save(user, "Create Group");
    const expected: RoutingGroup = {
      group_name: "probe-group",
      models: ["gpt-4o"],
      routing_strategy: "latency-based-routing",
      routing_strategy_args: null,
    };
    expect(onSubmit.mock.calls[0][0]).toStrictEqual(expected);
  });

  it("blocks a model another group already claims", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal({ groupNameByModel: { "gpt-4o": "cheap" } });

    await typeName(user, "security");
    await pickModels(user, "gpt-4o");
    await pickStrategy(user, "latency-based-routing");
    await save(user, "Create Group");

    expect(await screen.findByText(/Already claimed: gpt-4o/)).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("describes the selected strategy", async () => {
    renderModal();

    expect(await screen.findByText("Spreads requests evenly across the group.")).toBeInTheDocument();
  });
});
