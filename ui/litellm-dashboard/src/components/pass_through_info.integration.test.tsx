import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "../../tests/test-utils";
import PassThroughInfoView from "./pass_through_info";

const updatePassThroughEndpoint = vi.fn();

vi.mock("./networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./networking")>()),
  updatePassThroughEndpoint: (...args: unknown[]) => updatePassThroughEndpoint(...args),
  deletePassThroughEndpointsCall: vi.fn().mockResolvedValue({}),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn(), error: vi.fn() },
}));

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
type User = ReturnType<typeof setup>;

const endpoint = {
  id: "ep-1",
  path: "/bria",
  target: "https://engine.prod.bria-api.com",
  headers: { Authorization: "Bearer abc" },
  include_subpath: true,
  cost_per_request: 2,
  timeout: 600,
  auth: false,
  methods: ["GET"],
};

const renderView = (premiumUser = true, data = endpoint) =>
  renderWithProviders(
    <PassThroughInfoView
      endpointData={data}
      onClose={vi.fn()}
      accessToken="test-token"
      isAdmin
      premiumUser={premiumUser}
    />,
  );

const openEditForm = async (user: User) => {
  await user.click(screen.getByRole("tab", { name: "Settings" }));
  await user.click(await screen.findByRole("button", { name: "Edit Settings" }));
  await screen.findByLabelText("Target URL");
};

const save = async (user: User) => user.click(screen.getByRole("button", { name: "Save Changes" }));

const lastPayload = () => updatePassThroughEndpoint.mock.calls.at(-1)?.[2] as Record<string, unknown>;

describe("pass_through_info update payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    updatePassThroughEndpoint.mockResolvedValue({});
  });

  it("sends exactly the antd payload, once, for an untouched save", async () => {
    const user = setup();
    renderView();
    await openEditForm(user);

    await save(user);

    await waitFor(() => expect(updatePassThroughEndpoint).toHaveBeenCalled());

    expect(updatePassThroughEndpoint).toHaveBeenCalledTimes(1);
    expect(updatePassThroughEndpoint.mock.calls[0][0]).toBe("test-token");
    expect(updatePassThroughEndpoint.mock.calls[0][1]).toBe("ep-1");
    expect(lastPayload()).toStrictEqual({
      path: "/bria",
      target: "https://engine.prod.bria-api.com",
      headers: { Authorization: "Bearer abc" },
      include_subpath: true,
      cost_per_request: 2,
      timeout: 600,
      auth: false,
      methods: ["GET"],
      guardrails: undefined,
    });
  });

  it("submits numeric fields as numbers, not strings", async () => {
    const user = setup();
    renderView();
    await openEditForm(user);

    await save(user);

    await waitFor(() => expect(updatePassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().cost_per_request).toBe(2);
    expect(lastPayload().timeout).toBe(600);
  });

  it("keeps the path from the loaded endpoint rather than the form", async () => {
    const user = setup();
    renderView();
    await openEditForm(user);

    await user.clear(screen.getByLabelText("Target URL"));
    await user.type(screen.getByLabelText("Target URL"), "https://new.example.com");
    await save(user);

    await waitFor(() => expect(updatePassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().path).toBe("/bria");
    expect(lastPayload().target).toBe("https://new.example.com");
  });

  it("parses the headers textarea from JSON into an object", async () => {
    const user = setup();
    renderView();
    await openEditForm(user);

    await user.clear(screen.getByLabelText("Headers (JSON)"));
    await user.type(screen.getByLabelText("Headers (JSON)"), '{{"x-api-key": "k"}');
    await save(user);

    await waitFor(() => expect(updatePassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().headers).toStrictEqual({ "x-api-key": "k" });
  });

  it("sends auth undefined for a non-premium user", async () => {
    const user = setup();
    renderView(false);
    await openEditForm(user);

    await save(user);

    await waitFor(() => expect(updatePassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().auth).toBeUndefined();
  });

  it("sends methods undefined when the endpoint has none", async () => {
    const user = setup();
    renderView(true, { ...endpoint, methods: [] });
    await openEditForm(user);

    await save(user);

    await waitFor(() => expect(updatePassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().methods).toBeUndefined();
  });

  it("rounds cost_per_request to two decimals", async () => {
    const user = setup();
    renderView();
    await openEditForm(user);

    await user.clear(screen.getByLabelText("Cost per Request"));
    await user.type(screen.getByLabelText("Cost per Request"), "1.239");
    await save(user);

    await waitFor(() => expect(updatePassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().cost_per_request).toBe(1.24);
  });

  it("does not send when the target is cleared", async () => {
    const user = setup();
    renderView();
    await openEditForm(user);

    await user.clear(screen.getByLabelText("Target URL"));
    await save(user);

    expect(await screen.findByText("Please input a target URL")).toBeInTheDocument();
    expect(updatePassThroughEndpoint).not.toHaveBeenCalled();
  });

  it("submits when Enter is pressed in the target field", async () => {
    const user = setup();
    renderView();
    await openEditForm(user);

    await user.type(screen.getByLabelText("Target URL"), "{Enter}");

    await waitFor(() => expect(updatePassThroughEndpoint).toHaveBeenCalled());
    expect(updatePassThroughEndpoint).toHaveBeenCalledTimes(1);
  });
});
