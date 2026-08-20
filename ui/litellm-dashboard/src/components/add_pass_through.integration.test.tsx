import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, waitFor } from "../../tests/test-utils";
import AddPassThroughEndpoint from "./add_pass_through";

const createPassThroughEndpoint = vi.fn();

vi.mock("./networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./networking")>()),
  createPassThroughEndpoint: (...args: unknown[]) => createPassThroughEndpoint(...args),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn(), error: vi.fn() },
}));

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

type User = ReturnType<typeof setup>;

const renderForm = (premiumUser = true) =>
  renderWithProviders(
    <AddPassThroughEndpoint
      accessToken="test-token"
      passThroughItems={[]}
      setPassThroughItems={vi.fn()}
      premiumUser={premiumUser}
    />,
  );

const openModal = async (user: User) => {
  await user.click(screen.getByRole("button", { name: "+ Add Pass-Through Endpoint" }));
  await screen.findByText("Route Configuration");
};

interface RequiredFieldValues {
  path?: string;
  target?: string;
  headerName?: string;
  headerValue?: string;
}

const fillRequiredFields = async (user: User, values: RequiredFieldValues = {}) => {
  const {
    path = "bria",
    target = "https://example.com",
    headerName = "Authorization",
    headerValue = "Bearer abc",
  } = values;

  fireEvent.change(screen.getByPlaceholderText("bria"), { target: { value: path } });
  fireEvent.change(screen.getByPlaceholderText("https://engine.prod.bria-api.com"), {
    target: { value: target },
  });
  await user.click(screen.getByRole("button", { name: /add header/i }));
  fireEvent.change(screen.getByPlaceholderText("Header Name"), { target: { value: headerName } });
  fireEvent.change(screen.getByPlaceholderText("Header Value"), { target: { value: headerValue } });
};

const addQueryParam = async (user: User, name: string, value: string) => {
  await user.click(screen.getByRole("button", { name: /add query parameter/i }));
  fireEvent.change(screen.getByPlaceholderText("Parameter Name (e.g., version)"), { target: { value: name } });
  fireEvent.change(screen.getByPlaceholderText("Parameter Value (e.g., v1)"), { target: { value } });
};

const submit = async (user: User) => user.click(screen.getByRole("button", { name: "Add Pass-Through Endpoint" }));

const lastPayload = () => createPassThroughEndpoint.mock.calls.at(-1)?.[1] as Record<string, unknown>;

const reopenedFields: RequiredFieldValues = {
  path: "adobe",
  target: "https://adobe.example.com",
  headerName: "x-api-key",
  headerValue: "second-secret",
};

const reopenedPayload = {
  path: "/adobe",
  target: "https://adobe.example.com",
  headers: { "x-api-key": "second-secret" },
  include_subpath: true,
  methods: undefined,
  default_query_params: undefined,
  auth: undefined,
  timeout: undefined,
  cost_per_request: undefined,
};

describe("add_pass_through submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createPassThroughEndpoint.mockResolvedValue({ endpoints: [{ id: "generated-id", path: "/bria" }] });
  });

  it("sends exactly the antd payload, once, for a minimal endpoint", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);
    fireEvent.change(screen.getByPlaceholderText("600"), { target: { value: "900" } });
    fireEvent.change(screen.getByPlaceholderText("2.0000"), { target: { value: "1.5" } });

    await submit(user);

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());

    expect(createPassThroughEndpoint).toHaveBeenCalledTimes(1);
    expect(createPassThroughEndpoint.mock.calls[0][0]).toBe("test-token");
    expect(lastPayload()).toStrictEqual({
      path: "/bria",
      target: "https://example.com",
      headers: { Authorization: "Bearer abc" },
      include_subpath: true,
      methods: undefined,
      default_query_params: undefined,
      auth: undefined,
      timeout: "900",
      cost_per_request: "1.5",
    });
  });

  it("submits numeric fields as strings, not numbers", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);
    fireEvent.change(screen.getByPlaceholderText("600"), { target: { value: "900" } });
    fireEvent.change(screen.getByPlaceholderText("2.0000"), { target: { value: "1.5" } });

    await submit(user);

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().timeout).toBe("900");
    expect(lastPayload().cost_per_request).toBe("1.5");
  });

  it("prefixes the path with a slash when the user omits it", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);

    await submit(user);

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().path).toBe("/bria");
  });

  it("includes selected HTTP methods as an array", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);

    await user.click(screen.getByLabelText(/HTTP Methods/));
    await user.click(await screen.findByTitle("POST"));

    await submit(user);

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().methods).toStrictEqual(["POST"]);
  });

  it("sends auth true once the premium security toggle is switched on", async () => {
    const user = setup();
    renderForm(true);
    await openModal(user);
    await fillRequiredFields(user);

    const switches = screen.getAllByRole("switch");
    expect(switches).toHaveLength(2);
    await user.click(switches[1]);

    await submit(user);

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().auth).toBe(true);
  });

  it("omits the auth key entirely for a non-premium user", async () => {
    const user = setup();
    renderForm(false);
    await openModal(user);
    await fillRequiredFields(user);

    await submit(user);

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload()).not.toHaveProperty("auth");
  });

  it("drops include_subpath to false when the toggle is switched off", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);

    await user.click(screen.getAllByRole("switch")[0]);

    await submit(user);

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());
    expect(lastPayload().include_subpath).toBe(false);
  });

  it("does not submit when required fields are missing", async () => {
    const user = setup();
    renderForm();
    await openModal(user);

    await submit(user);

    expect(await screen.findByText("Path is required")).toBeInTheDocument();
    expect(screen.getByText("Target URL is required")).toBeInTheDocument();
    expect(screen.getByText("Please configure the headers")).toBeInTheDocument();
    expect(createPassThroughEndpoint).not.toHaveBeenCalled();
  });

  it("does not submit when the target is not a valid URL", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);

    await user.clear(screen.getByPlaceholderText("https://engine.prod.bria-api.com"));
    fireEvent.change(screen.getByPlaceholderText("https://engine.prod.bria-api.com"), {
      target: { value: "not a url" },
    });

    await submit(user);

    expect(await screen.findByText("Please enter a valid URL")).toBeInTheDocument();
    expect(createPassThroughEndpoint).not.toHaveBeenCalled();
  });

  it("submits when Enter is pressed in the path field", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);

    await user.type(screen.getByPlaceholderText("bria"), "{Enter}");

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());
    expect(createPassThroughEndpoint).toHaveBeenCalledTimes(1);
    expect(lastPayload().path).toBe("/bria");
  });

  it("does not submit when Cancel is clicked", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(createPassThroughEndpoint).not.toHaveBeenCalled();
  });

  it("leaves no header or query parameter rows behind when the modal is cancelled and reopened", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);
    await addQueryParam(user, "version", "v1");

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await openModal(user);

    expect(screen.queryByPlaceholderText("Header Name")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Parameter Name (e.g., version)")).not.toBeInTheDocument();
  });

  it("submits the reopened endpoint instead of rejecting it as missing headers", async () => {
    const user = setup();
    renderForm();
    await openModal(user);
    await fillRequiredFields(user);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await openModal(user);
    await fillRequiredFields(user, reopenedFields);

    await submit(user);

    await waitFor(() => expect(createPassThroughEndpoint).toHaveBeenCalled());
    expect(screen.queryByText("Please configure the headers")).not.toBeInTheDocument();
    expect(lastPayload()).toStrictEqual(reopenedPayload);
  });
});
