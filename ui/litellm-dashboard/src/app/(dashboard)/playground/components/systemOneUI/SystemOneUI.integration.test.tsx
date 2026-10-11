import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render as rtlRender,
  screen,
  waitFor,
  waitForElementToBeRemoved,
  within,
} from "@testing-library/react";
import type { ReactElement } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SystemOneUI from "./SystemOneUI";
import type { SystemOneResponse } from "./lib/schemas";
import type { OpenAIDecisionsResponse } from "./lib/openAIDecisions";
import { curlPayload } from "./lib/curlCommand";

interface ModelGroupInfo {
  model_group: string;
  mode: string;
}

const DECISION_AND_CHAT_MODELS: ModelGroupInfo[] = [
  { model_group: "gpt-5.5", mode: "chat" },
  { model_group: "jev-latest", mode: "evaluation" },
  { model_group: "pplx-decider", mode: "evaluation" },
];

const responseBody: SystemOneResponse = {
  model: "jev-1.13.0",
  answers: {
    department: {
      type: "choice",
      choice: "technical",
      confidence: 0.9,
      probabilities: { technical: 0.9, billing: 0.1 },
    },
  },
};

const render = (ui: ReactElement) =>
  rtlRender(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { mutations: { retry: false } } })}>
      {ui}
    </QueryClientProvider>,
  );

const modelGroupInfoResponse = (groups: ModelGroupInfo[]) =>
  ({ ok: true, status: 200, text: async () => JSON.stringify({ data: groups }) }) as Response;

const EXPIRED_KEY_RESPONSE = {
  ok: false,
  status: 401,
  text: async () =>
    JSON.stringify({ error: { message: "Authentication Error - Expired Key.", type: "expired_key", code: "401" } }),
} as Response;

const isModelLookup = (input: RequestInfo | URL) => String(input).endsWith("/model_group/info");

const lookupAuthHeaders = (calls: Parameters<typeof fetch>[]) =>
  calls.map(([, init]) => Object.values((init?.headers ?? {}) as Record<string, string>));

const readCurlPayload = async () => {
  await userEvent.click(screen.getByRole("tab", { name: "cURL" }));
  const payload = curlPayload(screen.getByLabelText("cURL request").textContent ?? "");
  await userEvent.click(screen.getByRole("tab", { name: "Form" }));
  return JSON.parse(payload ?? "") as Record<string, unknown>;
};

const selectEndpoint = async (endpoint: "/v1/decisions" | "/v1/systemone") => {
  screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
  await userEvent.keyboard("{ArrowDown}");
  await userEvent.click(await screen.findByRole("option", { name: endpoint, exact: true }));
};

const selectPreset = async (label: string) => {
  screen.getByRole("combobox", { name: "Example" }).focus();
  await userEvent.keyboard("{ArrowDown}");
  await userEvent.click(await screen.findByRole("option", { name: label }));
};

const createResponse = (body: SystemOneResponse | OpenAIDecisionsResponse, status = 200, errorText = "") =>
  ({
    ok: status >= 200 && status < 300,
    status,
    text: async () => (status >= 200 && status < 300 ? JSON.stringify(body) : errorText),
  }) as Response;

describe("SystemOneUI integration", () => {
  const mockFetch = vi.fn<typeof fetch>();
  const mockModelLookup = vi.fn<typeof fetch>();

  beforeEach(() => {
    sessionStorage.clear();
    mockFetch.mockReset();
    mockModelLookup.mockReset();
    vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) =>
      isModelLookup(input) ? mockModelLookup(input, init) : mockFetch(input, init),
    );
    mockFetch.mockResolvedValue(createResponse(responseBody));
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse([]));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
    document.cookie = "token=; path=/; max-age=0";
  });

  it("switches between example presets and reads Custom once the request is edited", async () => {
    render(<SystemOneUI accessToken="session-key" />);
    const input = screen.getByRole("textbox", { name: "Input" }) as HTMLTextAreaElement;
    const example = input.value;
    const picker = screen.getByRole("combobox", { name: "Example" });
    expect(picker).toHaveTextContent("Triage a bug report");

    fireEvent.change(input, { target: { value: "Something else" } });
    expect(picker).toHaveTextContent("Custom");
    await selectPreset("Triage a bug report");
    expect(screen.getByRole("textbox", { name: "Input" })).toHaveValue(example);
    expect(picker).toHaveTextContent("Triage a bug report");

    await selectPreset("Score a sales lead");
    expect(picker).toHaveTextContent("Score a sales lead");
    expect(await readCurlPayload()).toMatchObject({ questions: { fit: { type: "score" } } });
  });

  it("clears the whole request and lets a preset be picked from nothing", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    const clear = screen.getByRole("button", { name: "Clear" });
    expect(screen.getByRole("textbox", { name: "Input" })).toHaveDisplayValue(/streaming responses/);

    await user.click(clear);
    expect(screen.getByRole("textbox", { name: "Input" })).toHaveValue("");
    expect(screen.queryAllByRole("group", { name: /Question / })).toHaveLength(0);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(clear).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Example" })).toHaveTextContent("Custom");

    await selectPreset("Route a support ticket");
    expect(screen.getByRole("textbox", { name: "Input" })).toHaveDisplayValue(/charged twice/);
    expect(screen.getAllByRole("group", { name: /Question / })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
  });

  it("keeps the preset label after picking a model and keeps that model when clearing", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken="session-key" />);
    const modelPicker = screen.getByRole("combobox", { name: "Decision model" });
    await waitFor(() => expect(modelPicker).toHaveValue("jev-latest"));
    const picker = screen.getByRole("combobox", { name: "Example" });

    await user.clear(modelPicker);
    await user.type(modelPicker, "p");
    await user.click(await screen.findByRole("option", { name: "pplx-decider" }));
    expect(picker).toHaveTextContent("Triage a bug report");
    expect(await readCurlPayload()).toMatchObject({ model: "pplx-decider" });

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(await readCurlPayload()).toEqual({ model: "pplx-decider", state: "", questions: {} });
    await selectPreset("Score a sales lead");
    expect(await readCurlPayload()).toMatchObject({ model: "pplx-decider", questions: { fit: { type: "score" } } });
  });

  it("offers the same presets on /v1/decisions in the OpenAI shape and clears to an empty input", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    await selectEndpoint("/v1/decisions");
    await selectPreset("Moderate a forum comment");
    const payload = await readCurlPayload();
    expect(payload.input).toMatch(/stream=false/);
    expect(payload.questions).toEqual([
      { type: "predicate", name: "breaks_rules", instructions: "Does this comment break the community guidelines?" },
    ]);

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(await readCurlPayload()).toEqual({ input: "", questions: [] });
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("offers both literal proxy endpoints without TypeSafe or product prefixes", async () => {
    render(<SystemOneUI accessToken="session-key" />);

    expect(screen.getByRole("combobox", { name: "Decision endpoint" })).toHaveTextContent("/v1/systemone");
    screen.getByRole("combobox", { name: "Decision endpoint" }).focus();
    await userEvent.keyboard("{ArrowDown}");
    expect(await screen.findByRole("option", { name: "/v1/decisions", exact: true })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "/v1/systemone", exact: true })).toBeInTheDocument();
    expect(screen.getAllByRole("option")).toHaveLength(2);
    expect(screen.queryByText(/TypeSafe/)).not.toBeInTheDocument();
    await userEvent.keyboard("{Escape}");

    expect(screen.getByRole("link", { name: "How to call /v1/decisions and /v1/systemone" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/decisions",
    );
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("opens on /v1/systemone with the first decision model the key can call", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken="session-key" />);

    await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));
    expect(lookupAuthHeaders(mockModelLookup.mock.calls)).toEqual([expect.arrayContaining(["Bearer session-key"])]);
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(mockFetch.mock.calls[0]?.[0]).not.toMatch(/typesafe/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).toMatchObject({
      model: "jev-latest",
      questions: { has_repro_steps: { type: "noul" } },
    });
  });

  it("sends each endpoint's real shape and switches existing responses between cards and JSON without resending", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken="session-key" />);
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));
    fireEvent.change(screen.getByRole("textbox", { name: "Input" }), { target: { value: "An outage" } });

    await selectEndpoint("/v1/decisions");
    expect(screen.getByRole("textbox", { name: "Input" })).not.toHaveValue("An outage");
    expect(await readCurlPayload()).toMatchObject({
      input: expect.any(String),
      questions: expect.arrayContaining([expect.objectContaining({ type: "predicate" })]),
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Input" }), { target: { value: "An outage" } });
    const response: OpenAIDecisionsResponse = {
      model: "jev-latest",
      answers: [{ name: "has_repro_steps", type: "predicate", probability: 0.95 }],
      usage: { input_tokens: 10, output_tokens: 1, total_tokens: 11 },
    };
    mockFetch.mockResolvedValueOnce(createResponse(response));
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("meter", { name: "Yes probability" })).toHaveAttribute("aria-valuenow", "95");
    expect(screen.queryByRole("region", { name: "Decisions response JSON" })).not.toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/decisions$/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).toMatchObject({
      model: "jev-latest",
      input: "An outage",
      questions: expect.arrayContaining([expect.objectContaining({ type: "predicate" })]),
    });

    await user.click(screen.getByRole("tab", { name: "cURL" }));
    const result = screen.getByRole("region", { name: "Decisions response JSON" });
    expect(result).toHaveTextContent('"probability": 0.95');
    expect(result).toHaveTextContent('"answers": [');
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Form" }));
    expect(screen.getByRole("meter", { name: "Yes probability" })).toBeInTheDocument();
    expect(mockFetch).toHaveBeenCalledTimes(1);

    await selectEndpoint("/v1/systemone");
    expect(screen.getByRole("textbox", { name: "Input" })).toHaveValue("An outage");
    expect(screen.queryByRole("region", { name: "Decisions results" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "technical probability" })).toHaveAttribute("aria-valuenow", "90");
    await user.click(screen.getByRole("tab", { name: "cURL" }));
    expect(screen.getByRole("region", { name: "Decisions response JSON" })).toHaveTextContent('"choice": "technical"');
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(mockFetch.mock.calls[1]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(JSON.parse(mockFetch.mock.calls[1]?.[1]?.body as string)).toMatchObject({
      model: "jev-latest",
      state: "An outage",
    });

    await selectEndpoint("/v1/decisions");
    expect(await readCurlPayload()).toMatchObject({ input: "An outage" });
    await selectPreset("Triage a bug report");
    expect(screen.getByRole("textbox", { name: "Input" })).not.toHaveValue("An outage");
    await selectEndpoint("/v1/systemone");
    expect(screen.getByRole("textbox", { name: "Input" })).toHaveValue("An outage");
  });

  it("validates /v1/decisions independently and preserves an invalid draft across switches", async () => {
    render(<SystemOneUI accessToken="session-key" />);
    await selectEndpoint("/v1/decisions");
    await userEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    await selectEndpoint("/v1/systemone");
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
    await selectEndpoint("/v1/decisions");
    expect(screen.getByRole("textbox", { name: "Input" })).toHaveValue("");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("offers a Decisions form that sends input and typed questions and mirrors them in cURL", async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce(
      createResponse({
        model: "jev-latest",
        answers: [],
        usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
      }),
    );
    render(<SystemOneUI accessToken="session-key" />);
    await selectEndpoint("/v1/decisions");
    expect(screen.getByRole("tab", { name: "Form" })).toHaveAttribute("aria-selected", "true");
    fireEvent.change(screen.getByRole("textbox", { name: "Input" }), { target: { value: "Login is broken" } });
    const firstQuestion = within(screen.getByRole("group", { name: "Question 1" }));
    fireEvent.change(firstQuestion.getByRole("textbox", { name: "Question name" }), { target: { value: "team" } });
    fireEvent.change(firstQuestion.getByRole("textbox", { name: "Instructions" }), {
      target: { value: "Which team owns this?" },
    });
    fireEvent.change(firstQuestion.getByRole("textbox", { name: "Choice 1 value" }), {
      target: { value: "infra" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Safety identifier" }), {
      target: { value: "qa-session" },
    });
    const payload = (await readCurlPayload()) as { questions: unknown[] };
    expect(payload).toMatchObject({ input: "Login is broken", safety_identifier: "qa-session" });
    const expectedQuestion = {
      name: "team",
      type: "choice",
      instructions: "Which team owns this?",
      choices: expect.arrayContaining([expect.objectContaining({ value: "infra" })]),
    };
    expect(payload.questions[0]).toMatchObject(expectedQuestion);
    expect(payload).not.toHaveProperty("state");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(1));
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/decisions$/);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).toEqual(payload);
    expect(await screen.findByRole("button", { name: "Raw response" })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "cURL" }));
    expect(screen.getByRole("region", { name: "Decisions response JSON" })).toBeInTheDocument();
  });

  it("adds and removes Decisions questions, boolean choices and score levels without stale fields", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    await selectEndpoint("/v1/decisions");
    const question = (index: number) => within(screen.getByRole("group", { name: `Question ${index}` }));
    const pick = async (control: HTMLElement, label: string) => {
      control.focus();
      await user.keyboard("{ArrowDown}");
      await user.click(await screen.findByRole("option", { name: label, exact: true }));
    };
    await user.click(screen.getByRole("button", { name: "Add question" }));
    fireEvent.change(question(4).getByRole("textbox", { name: "Instructions" }), {
      target: { value: "Should we escalate?" },
    });
    await pick(question(4).getByRole("combobox", { name: "Answer type" }), "Choice");
    await user.click(question(4).getByRole("button", { name: "Remove choice 1" }));
    expect(question(4).getByRole("textbox", { name: "Choice 1 value" })).toHaveValue("option_2");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    await user.click(question(4).getByRole("button", { name: "Add choice" }));
    await pick(question(4).getByRole("combobox", { name: "Choice 1 value type" }), "false (boolean)");
    fireEvent.change(question(4).getByRole("textbox", { name: "Choice 1 description" }), {
      target: { value: "Not urgent" },
    });
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
    const choicePayload = (await readCurlPayload()) as { questions: Record<string, unknown>[] };
    const expectedChoice = {
      type: "choice",
      name: "question_1",
      instructions: "Should we escalate?",
      choices: [{ value: false, description: "Not urgent" }, { value: "option_1" }],
    };
    expect(choicePayload.questions[3]).toEqual(expectedChoice);
    await user.click(screen.getByRole("tab", { name: "Form" }));
    await pick(question(4).getByRole("combobox", { name: "Answer type" }), "Score");
    fireEvent.change(question(4).getByRole("textbox", { name: "Level 1 label" }), { target: { value: "low" } });
    fireEvent.change(question(4).getByRole("textbox", { name: "Level 1 description" }), {
      target: { value: "No impact" },
    });
    await user.click(question(4).getByRole("button", { name: "Add level" }));
    await user.click(question(4).getByRole("button", { name: "Remove level 2" }));
    expect(question(4).getByRole("textbox", { name: "Level 2 label" })).toHaveValue("2");
    await user.click(question(2).getByRole("button", { name: "Remove question 2" }));
    expect(question(3).getByRole("textbox", { name: "Instructions" })).toHaveValue("Should we escalate?");
    const scorePayload = (await readCurlPayload()) as { questions: Record<string, unknown>[] };
    expect(scorePayload.questions).toHaveLength(3);
    const expectedScore = {
      type: "score",
      name: "question_1",
      instructions: "Should we escalate?",
      levels: [{ label: "low", description: "No impact" }, { label: "2" }],
    };
    expect(scorePayload.questions[2]).toEqual(expectedScore);
    await user.click(screen.getByRole("tab", { name: "Form" }));
    await pick(question(3).getByRole("combobox", { name: "Answer type" }), "Predicate");
    const predicatePayload = (await readCurlPayload()) as { questions: Record<string, unknown>[] };
    expect(predicatePayload.questions[2]).toEqual({
      type: "predicate",
      name: "question_1",
      instructions: "Should we escalate?",
    });
  });

  it.each(["success", "error"])("cancels the previous endpoint request and ignores a late %s", async (outcome) => {
    const pending = Promise.withResolvers<Response>();
    mockFetch.mockReturnValueOnce(pending.promise);
    const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    rtlRender(
      <QueryClientProvider client={queryClient}>
        <SystemOneUI accessToken="session-key" />
      </QueryClientProvider>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByRole("button", { name: "Cancel request" });
    await selectEndpoint("/v1/decisions");
    expect(mockFetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    await act(async () => {
      if (outcome === "success") {
        pending.resolve(createResponse({ model: "stale-model", answers: {} }));
      } else {
        pending.reject(new Error("Stale request failed"));
      }
      await pending.promise.catch(() => undefined);
    });
    await waitFor(() => expect(queryClient.isMutating()).toBe(0));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText("stale-model")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Form" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
  });

  it("offers only decision models and writes the picked one into the request", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken="session-key" />);
    const picker = await screen.findByRole("combobox", { name: "Decision model" });

    await user.clear(picker);
    await user.type(picker, "p");
    expect(await screen.findByRole("option", { name: "pplx-decider" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "gpt-5.5" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: "pplx-decider" }));

    expect(await readCurlPayload()).toMatchObject({
      model: "pplx-decider",
      questions: { has_repro_steps: { type: "noul" } },
    });
    await user.click(screen.getByRole("button", { name: "Send" }));
    await user.click(screen.getByRole("tab", { name: "Form" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string).model).toBe("pplx-decider");
  });

  it("routes a configured TypeSafe model through the same proxy endpoint", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(
      modelGroupInfoResponse([{ model_group: "typesafe/jev-latest", mode: "evaluation" }]),
    );
    render(<SystemOneUI accessToken="session-key" />);
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("typesafe/jev-latest"),
    );

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(mockFetch.mock.calls[0]?.[0]).not.toMatch(/\/typesafe\//);
    expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string).model).toBe("typesafe/jev-latest");
  });

  it("links to Add Model when the key cannot call any decision model", async () => {
    render(<SystemOneUI accessToken="session-key" />);

    expect(await screen.findByRole("link", { name: "Add a decision model" })).toHaveAttribute(
      "href",
      "/ui/models-and-endpoints",
    );
    expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveAttribute(
      "placeholder",
      "No decision models yet",
    );
  });

  it("looks up a typed virtual key's decision models once, after typing settles", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    render(<SystemOneUI accessToken={null} disabledPersonalKeyCreation />);

    await user.type(screen.getByLabelText("Virtual Key", { exact: true }), "sk-typed-key");

    await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));
    expect(lookupAuthHeaders(mockModelLookup.mock.calls)).toEqual([expect.arrayContaining(["Bearer sk-typed-key"])]);
  });

  it("keeps the admin signed in when a typed virtual key has expired", async () => {
    document.cookie = "token=ui-session; path=/";
    mockModelLookup.mockResolvedValue(EXPIRED_KEY_RESPONSE);
    render(<SystemOneUI accessToken={null} disabledPersonalKeyCreation />);

    fireEvent.change(screen.getByLabelText("Virtual Key", { exact: true }), { target: { value: "sk-expired" } });

    await waitFor(() => expect(mockModelLookup).toHaveBeenCalled());
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(document.cookie).toContain("token=ui-session");
    expect(screen.getByRole("combobox", { name: "Decision endpoint" })).toHaveTextContent("/v1/systemone");
  });

  it("keeps an edited request in place when decision models load afterwards", async () => {
    const models = Promise.withResolvers<Response>();
    mockModelLookup.mockReturnValue(models.promise);
    render(<SystemOneUI accessToken="session-key" />);
    fireEvent.change(screen.getByRole("textbox", { name: "Input" }), { target: { value: "edited" } });

    await act(async () => {
      models.resolve(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(screen.getByRole("textbox", { name: "Input" })).toHaveValue("edited");
    expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("");
  });

  it.each(["/v1/decisions", "/v1/systemone"] as const)(
    "omits model while discovery is pending on %s",
    async (endpoint) => {
      const models = Promise.withResolvers<Response>();
      mockModelLookup.mockReturnValue(models.promise);
      mockFetch.mockResolvedValueOnce(
        createResponse({
          model: "jev-latest",
          answers: endpoint === "/v1/decisions" ? [] : {},
          usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        }),
      );
      render(<SystemOneUI accessToken="session-key" />);
      if (endpoint === "/v1/decisions") {
        await selectEndpoint(endpoint);
      }
      await userEvent.click(screen.getByRole("button", { name: "Send" }));
      expect(await screen.findByText("jev-latest")).toBeInTheDocument();
      expect(mockFetch.mock.calls[0]?.[0]).toMatch(new RegExp(`${endpoint}$`));
      expect(JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string)).not.toHaveProperty("model");
      await act(async () => {
        models.resolve(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
        await models.promise;
      });
      expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("");
    },
  );

  it("shows a read-only cURL command for the proxy with a key placeholder instead of the real key", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem("customProxyBaseUrl", "https://proxy.example/");
    render(<SystemOneUI accessToken="session-key" />);
    await user.click(screen.getByRole("tab", { name: "cURL" }));

    const command = screen.getByLabelText("cURL request");
    expect(command).toHaveTextContent("curl -X POST 'https://proxy.example/v1/systemone'");
    expect(command).toHaveTextContent("Authorization: Bearer $LITELLM_API_KEY");
    expect(command).not.toHaveTextContent("session-key");
    expect(JSON.parse(curlPayload(command.textContent ?? "") ?? "")).toMatchObject({
      questions: { has_repro_steps: { type: "noul" } },
    });
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy cURL" })).toBeInTheDocument();

    await selectEndpoint("/v1/decisions");
    expect(screen.getByLabelText("cURL request")).toHaveTextContent("https://proxy.example/v1/decisions");
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
  });

  it("keeps Send below the form and shows results only after a request", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);

    const send = screen.getByRole("button", { name: "Send" });
    const editor = screen.getByRole("region", { name: "Decisions request editor" });
    const form = within(editor).getByRole("textbox", { name: "Input" });
    expect(form.compareDocumentPosition(send) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Decisions results" })).not.toBeInTheDocument();
    expect(screen.queryByRole("note")).not.toBeInTheDocument();

    await user.click(send);

    const results = await screen.findByRole("region", { name: "Decisions results" });
    expect(within(results).getByText("Calibrated probabilities")).toBeInTheDocument();
    expect(editor.compareDocumentPosition(results) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    fireEvent.change(within(editor).getByRole("textbox", { name: "Input" }), { target: { value: "Edited state" } });
    expect(screen.queryByRole("region", { name: "Decisions results" })).not.toBeInTheDocument();
  });

  it("ties each setup label to its control, numbers question cards and keeps the actions in the editor footer", () => {
    render(<SystemOneUI accessToken="session-key" />);

    expect(screen.getByText("Endpoint")).toHaveAttribute("for", screen.getByLabelText("Decision endpoint").id);
    expect(screen.getByText("Model")).toHaveAttribute("for", screen.getByLabelText("Decision model").id);
    expect(screen.getByText("API key")).toHaveAttribute("for", screen.getByLabelText("Virtual Key Source").id);

    const question = screen.getByRole("group", { name: "Question area" });
    expect(within(question).getByText("Question 1")).toBeInTheDocument();
    expect(within(question).getByText("Name")).toBeInTheDocument();
    expect(within(question).getByText("Answer type")).toBeInTheDocument();

    const editor = screen.getByRole("region", { name: "Decisions request editor" });
    const status = within(editor).getByText("Ready to send");
    const actions = within(editor).getByRole("toolbar", { name: "Request actions" });
    const clear = within(actions).getByRole("button", { name: "Clear" });
    const send = within(actions).getByRole("button", { name: "Send" });
    expect(status.compareDocumentPosition(actions) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(clear.compareDocumentPosition(send) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("sends to /v1/systemone even without discovered models and renders calibrated answers", async () => {
    const user = userEvent.setup();
    sessionStorage.setItem("customProxyBaseUrl", "https://stale.example.com/");
    render(<SystemOneUI accessToken="session-key" />);
    sessionStorage.setItem("customProxyBaseUrl", "https://tenant.example.com/");

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findAllByText("technical")).toHaveLength(2);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, options] = mockFetch.mock.calls[0] ?? [];
    expect(url).toBe("https://tenant.example.com/v1/systemone");
    expect(options?.method).toBe("POST");
    expect(options?.headers).toMatchObject({ "Content-Type": "application/json" });
    expect(Object.values(options?.headers as Record<string, string>)).toContain("Bearer session-key");
    expect(JSON.parse(options?.body as string)).toMatchObject({
      questions: { has_repro_steps: { type: "noul" } },
    });
    expect(JSON.parse(options?.body as string)).not.toHaveProperty("model");
  });

  it("uses the requested model when the response omits it", async () => {
    const user = userEvent.setup();
    mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
    mockFetch.mockResolvedValueOnce(
      createResponse({
        answers: responseBody.answers,
      }),
    );
    render(<SystemOneUI accessToken="session-key" />);

    await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("jev-latest")).toBeInTheDocument();
    expect(screen.getByText("Selected choice")).toBeInTheDocument();
  });

  it("sends a native request without model so the proxy can select its default", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);
    fireEvent.change(screen.getByRole("textbox", { name: "Input" }), { target: { value: "An outage" } });
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("jev-1.13.0")).toBeInTheDocument();
    expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
    const body = JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string);
    expect(body).toMatchObject({ state: "An outage" });
    expect(body).not.toHaveProperty("model");
  });

  it.each(["success", "error"])("ignores a late %s after editing the request", async (outcome) => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Response>();
    mockFetch.mockReturnValueOnce(pending.promise);
    const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    rtlRender(
      <QueryClientProvider client={queryClient}>
        <SystemOneUI accessToken="session-key" />
      </QueryClientProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByRole("button", { name: "Cancel request" });
    fireEvent.change(screen.getByRole("textbox", { name: "Input" }), { target: { value: "A new request" } });
    expect(mockFetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();

    await act(async () => {
      if (outcome === "success") {
        pending.resolve(createResponse({ model: "stale-model", answers: { stale: { type: "noul", noul: 1 } } }));
      } else {
        pending.reject(new Error("Stale request failed"));
      }
      await pending.promise.catch(() => undefined);
    });
    await waitFor(() => expect(queryClient.isMutating()).toBe(0));
    expect(screen.queryByText("stale-model")).not.toBeInTheDocument();
    expect(screen.queryByText("Stale request failed")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText("100% yes")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("jev-1.13.0")).toBeInTheDocument();
    expect(screen.getByText("Selected choice")).toBeInTheDocument();
    expect(mockFetch.mock.calls[1]?.[0]).toMatch(/\/v1\/systemone$/);
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("uses a custom virtual key when personal key creation is disabled", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken={null} disabledPersonalKeyCreation />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Virtual Key", { exact: true }), {
      target: { value: "test-virtual-key" },
    });
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();
    expect(Object.values(mockFetch.mock.calls[0]?.[1]?.headers ?? {})).toContain("Bearer test-virtual-key");
  });

  it("renders upstream errors inline", async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce(createResponse(responseBody, 401, "Virtual key rejected"));
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Virtual key rejected");
  });

  it("renders malformed success responses inline", async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () =>
        JSON.stringify({
          model: "jev-1.13.0",
          answers: {
            department: {
              type: "choice",
              choice: "technical",
              probabilities: null,
            },
          },
        }),
    } as Response);
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("/v1/systemone response has an invalid shape.");
    expect(screen.queryByText("jev-1.13.0")).not.toBeInTheDocument();
  });

  it("clears a previous answer when the request editor changes", async () => {
    const user = userEvent.setup();
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "Input" }), {
      target: { value: "A different support request" },
    });

    expect(screen.queryByText("Selected choice")).not.toBeInTheDocument();
    expect(screen.queryByText("jev-1.13.0")).not.toBeInTheDocument();
  });

  it("clears the answer when the UI session key changes", async () => {
    const user = userEvent.setup();
    const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    const view = rtlRender(
      <QueryClientProvider client={queryClient}>
        <SystemOneUI accessToken="session-key" />
      </QueryClientProvider>,
    );

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Selected choice")).toBeInTheDocument();

    view.rerender(
      <QueryClientProvider client={queryClient}>
        <SystemOneUI accessToken="new-session-key" />
      </QueryClientProvider>,
    );

    await waitForElementToBeRemoved(() => screen.queryByText("Selected choice"));
  });

  it("aborts an in-flight request when cancelled", async () => {
    const user = userEvent.setup();
    mockFetch.mockImplementation(
      (_input, init) =>
        new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("Request aborted", "AbortError")), {
            once: true,
          });
        }),
    );
    render(<SystemOneUI accessToken="session-key" />);

    await user.click(screen.getByRole("button", { name: "Send" }));
    await user.click(await screen.findByRole("button", { name: "Cancel request" }));

    expect(mockFetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
  });

  describe("request form", () => {
    const question = (name: string) => within(screen.getByRole("group", { name: `Question ${name}` }));
    const sentBody = () => JSON.parse(mockFetch.mock.calls[0]?.[1]?.body as string);

    async function pickAnswerType(user: ReturnType<typeof userEvent.setup>, name: string, type: string) {
      question(name).getByRole("combobox", { name: "Answer type" }).focus();
      await user.keyboard("{ArrowDown}");
      await user.click(await screen.findByRole("option", { name: type }));
    }

    const fill = (field: HTMLElement, value: string) => fireEvent.change(field, { target: { value } });

    function rename(field: HTMLElement, name: string) {
      fill(field, name);
      fireEvent.blur(field);
    }

    it("explains the yes / no answer type option", async () => {
      const user = userEvent.setup();
      render(<SystemOneUI accessToken="session-key" />);

      question("has_repro_steps").getByRole("combobox", { name: "Answer type" }).focus();
      await user.keyboard("{ArrowDown}");
      expect(await screen.findByRole("option", { name: "Yes / no" })).toHaveAttribute(
        "title",
        "The model returns the probability that the answer is yes",
      );
      expect(screen.getByRole("option", { name: "Choice" })).not.toHaveAttribute("title");
    });

    it("sends a choice and a yes/no question built in the form without opening JSON", async () => {
      const user = userEvent.setup();
      render(<SystemOneUI accessToken="session-key" />);
      const input = screen.getByRole("textbox", { name: "Input" });
      fill(input, "The login page is blank after the upgrade");
      for (const name of ["area", "has_repro_steps", "severity"]) {
        await user.click(screen.getByRole("button", { name: `Remove question ${name}` }));
      }

      await user.click(screen.getByRole("button", { name: "Add question" }));
      rename(question("question_1").getByRole("textbox", { name: "Question name" }), "team");
      fill(question("team").getByRole("textbox", { name: "Instructions" }), "Which team owns this?");
      const [first, second] = question("team").getAllByRole("textbox", { name: "Option label" });
      rename(first, "frontend");
      rename(second, "backend");
      fill(question("team").getByRole("textbox", { name: "Description of frontend" }), "Dashboard and UI");
      await user.click(question("team").getByRole("button", { name: "Add option" }));
      rename(question("team").getAllByRole("textbox", { name: "Option label" })[2], "infra");
      fill(question("team").getByRole("textbox", { name: "Description of infra" }), "Deploys");

      await user.click(screen.getByRole("button", { name: "Add question" }));
      await pickAnswerType(user, "question_1", "Yes / no");
      rename(question("question_1").getByRole("textbox", { name: "Question name" }), "urgent");
      fill(question("urgent").getByRole("textbox", { name: "Instructions" }), "Is this blocking users?");
      fill(question("urgent").getByRole("textbox", { name: "Yes means" }), "Nobody can log in");

      await user.click(screen.getByRole("button", { name: "Send" }));

      expect(await screen.findByText("Selected choice")).toBeInTheDocument();
      expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
      expect(sentBody()).toEqual({
        state: "The login page is blank after the upgrade",
        questions: {
          team: {
            type: "choice",
            instructions: "Which team owns this?",
            criteria: { frontend: "Dashboard and UI", backend: "", infra: "Deploys" },
          },
          urgent: { type: "noul", instructions: "Is this blocking users?", criteria: { true: "Nobody can log in" } },
        },
      });
      expect(Object.keys(sentBody().questions)).toEqual(["team", "urgent"]);
      expect(Object.keys(sentBody().questions.team.criteria)).toEqual(["frontend", "backend", "infra"]);
    });

    it("edits score levels and keeps instructions when a question changes type", async () => {
      const user = userEvent.setup();
      mockModelLookup.mockResolvedValue(modelGroupInfoResponse(DECISION_AND_CHAT_MODELS));
      render(<SystemOneUI accessToken="session-key" />);
      await waitFor(() => expect(screen.getByRole("combobox", { name: "Decision model" })).toHaveValue("jev-latest"));

      const level = question("severity").getByRole("textbox", { name: "Level 0" });
      fill(level, "Typo");
      await user.click(question("severity").getByRole("button", { name: "Remove level 4" }));
      await user.click(question("severity").getByRole("button", { name: "Add level" }));
      fill(question("severity").getByRole("textbox", { name: "Level 4" }), "Data loss");
      await pickAnswerType(user, "has_repro_steps", "Choice");
      await user.click(question("has_repro_steps").getByRole("button", { name: "Add option" }));
      await user.click(question("area").getByRole("button", { name: "Remove option docs" }));

      await user.click(screen.getByRole("button", { name: "Send" }));

      expect(await screen.findByText("Selected choice")).toBeInTheDocument();
      expect(mockFetch.mock.calls[0]?.[0]).toMatch(/\/v1\/systemone$/);
      expect(sentBody().model).toBe("jev-latest");
      expect(sentBody().questions.severity.criteria).toEqual([
        "Typo",
        "Minor bug with an easy workaround",
        "Broken feature with a workaround",
        "Broken feature with no workaround",
        "Data loss",
      ]);
      expect(sentBody().questions.has_repro_steps).toEqual({
        type: "choice",
        instructions: "Does the issue include steps someone could follow to reproduce it?",
        criteria: { option_1: "", option_2: "", option_3: "" },
      });
      expect(Object.keys(sentBody().questions.area.criteria)).toEqual(["backend", "sdk", "ui"]);
    });

    it("refuses an empty or taken question name and trims a new one", async () => {
      const user = userEvent.setup();
      render(<SystemOneUI accessToken="session-key" />);
      const name = question("severity").getByRole("textbox", { name: "Question name" });
      expect(name).toHaveAttribute("aria-invalid", "false");

      fill(name, "area");
      expect(name).toHaveAttribute("aria-invalid", "true");
      expect(question("severity").getByText('"area" is already used')).toBeInTheDocument();
      fireEvent.blur(name);
      expect(name).toHaveValue("severity");

      fill(name, "");
      expect(question("severity").getByText("Enter a name")).toBeInTheDocument();
      fireEvent.blur(name);
      expect(name).toHaveValue("severity");

      rename(name, "  impact  ");
      expect(question("impact").getByRole("textbox", { name: "Question name" })).toHaveValue("impact");
      await user.click(screen.getByRole("button", { name: "Add question" }));
      await user.click(screen.getByRole("button", { name: "Add question" }));
      const json = (await readCurlPayload()) as { questions: Record<string, unknown> };
      expect(Object.keys(json.questions)).toEqual(["area", "has_repro_steps", "impact", "question_1", "question_2"]);
    });
  });
});
