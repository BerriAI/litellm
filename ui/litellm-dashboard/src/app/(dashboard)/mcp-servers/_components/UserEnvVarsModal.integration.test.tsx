import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import UserEnvVarsModal from "./UserEnvVarsModal";
import * as networking from "@/components/networking";
import { MCPServer, MCPUserEnvVarsStatus } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({
  getMCPUserEnvVars: vi.fn(),
  storeMCPUserEnvVars: vi.fn(),
}));

const createQueryClient = () => new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const server = { server_id: "srv-1", server_name: "Payments", alias: "payments" } as MCPServer;

const statusWith = (required: MCPUserEnvVarsStatus["required"]): MCPUserEnvVarsStatus =>
  ({ required }) as MCPUserEnvVarsStatus;

const renderModal = (status: MCPUserEnvVarsStatus, onSaved = vi.fn(), onClose = vi.fn()) => {
  vi.mocked(networking.getMCPUserEnvVars).mockResolvedValue(status);
  const view = render(
    <QueryClientProvider client={createQueryClient()}>
      <UserEnvVarsModal server={server} open accessToken="sk-test" onClose={onClose} onSaved={onSaved} />
    </QueryClientProvider>,
  );
  const setOpen = (open: boolean) =>
    view.rerender(
      <QueryClientProvider client={createQueryClient()}>
        <UserEnvVarsModal server={server} open={open} accessToken="sk-test" onClose={onClose} onSaved={onSaved} />
      </QueryClientProvider>,
    );
  return { onSaved, onClose, setOpen };
};

const save = (user: ReturnType<typeof setup>) => user.click(screen.getByRole("button", { name: "Save Credentials" }));

// Opening the modal remounts the form so nothing carries over from the last time it was open.
// Settle that remount before handing back a node, or the caller holds a detached one.
const fieldAfterOpen = async (label: RegExp): Promise<HTMLElement> => {
  await screen.findByLabelText(label);
  await act(async () => {});
  return screen.getByLabelText(label);
};

describe("UserEnvVarsModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("submits every declared field, trimmed, keyed by env var name", async () => {
    const user = setup();
    vi.mocked(networking.storeMCPUserEnvVars).mockResolvedValue(statusWith([]));
    renderModal(
      statusWith([
        { name: "API_KEY", description: "Your API key", is_set: false },
        { name: "REGION", description: null, is_set: false },
      ]),
    );

    fireEvent.change(await fieldAfterOpen(/^API_KEY/), { target: { value: "  secret-value  " } });
    fireEvent.change(screen.getByLabelText(/^REGION/), { target: { value: "us-east-1" } });
    await save(user);

    await waitFor(() => {
      expect(networking.storeMCPUserEnvVars).toHaveBeenCalledWith("sk-test", "srv-1", {
        API_KEY: "secret-value",
        REGION: "us-east-1",
      });
    });
    expect(networking.storeMCPUserEnvVars).toHaveBeenCalledTimes(1);
  });

  it("sends an empty string for an already-set field left blank", async () => {
    const user = setup();
    vi.mocked(networking.storeMCPUserEnvVars).mockResolvedValue(statusWith([]));
    renderModal(
      statusWith([
        { name: "API_KEY", description: null, is_set: true },
        { name: "REGION", description: null, is_set: true },
      ]),
    );

    fireEvent.change(await fieldAfterOpen(/^REGION/), { target: { value: "eu-west-2" } });
    await save(user);

    await waitFor(() => {
      expect(networking.storeMCPUserEnvVars).toHaveBeenCalledWith("sk-test", "srv-1", {
        API_KEY: "",
        REGION: "eu-west-2",
      });
    });
  });

  it("blocks the submit and shows the required message when an unset field is empty", async () => {
    const user = setup();
    renderModal(
      statusWith([
        { name: "API_KEY", description: null, is_set: false },
        { name: "REGION", description: null, is_set: true },
      ]),
    );

    await fieldAfterOpen(/^API_KEY/);
    await save(user);

    expect(await screen.findByText("API_KEY is required")).toBeInTheDocument();
    expect(networking.storeMCPUserEnvVars).not.toHaveBeenCalled();
  });

  it("does not require an already-set field", async () => {
    const user = setup();
    vi.mocked(networking.storeMCPUserEnvVars).mockResolvedValue(statusWith([]));
    renderModal(statusWith([{ name: "API_KEY", description: null, is_set: true }]));

    await fieldAfterOpen(/^API_KEY/);
    await save(user);

    await waitFor(() => {
      expect(networking.storeMCPUserEnvVars).toHaveBeenCalledWith("sk-test", "srv-1", { API_KEY: "" });
    });
  });

  it("renders the admin description as the placeholder for an unset field", async () => {
    renderModal(statusWith([{ name: "API_KEY", description: "Grab it from the console", is_set: false }]));

    expect(await screen.findByPlaceholderText("Grab it from the console")).toBeInTheDocument();
  });

  it("renders the overwrite placeholder and a Set marker for an already-set field", async () => {
    renderModal(statusWith([{ name: "API_KEY", description: "Grab it from the console", is_set: true }]));

    expect(await screen.findByPlaceholderText("Enter a new value to overwrite")).toBeInTheDocument();
    expect(screen.getByText("Set")).toBeInTheDocument();
  });

  it("renders the admin description as always-visible help text", async () => {
    renderModal(statusWith([{ name: "API_KEY", description: "Grab it from the console", is_set: true }]));

    await fieldAfterOpen(/^API_KEY/);
    expect(screen.getByText("Grab it from the console")).toBeVisible();
  });

  it("masks the entered value", async () => {
    const user = setup();
    renderModal(statusWith([{ name: "API_KEY", description: null, is_set: false }]));

    const input = await fieldAfterOpen(/^API_KEY/);
    expect(input).toHaveAttribute("type", "password");
    fireEvent.change(input, { target: { value: "hunter2" } });
    expect(screen.getByLabelText(/^API_KEY/)).toHaveAttribute("type", "password");
  });

  it("starts from a blank form each time it is opened", async () => {
    const { setOpen } = renderModal(statusWith([{ name: "API_KEY", description: null, is_set: false }]));

    const input = await fieldAfterOpen(/^API_KEY/);
    fireEvent.change(input, { target: { value: "hunter2" } });
    expect(screen.getByLabelText(/^API_KEY/)).toHaveValue("hunter2");

    setOpen(false);
    await waitFor(() => {
      expect(screen.queryByLabelText(/^API_KEY/)).not.toBeInTheDocument();
    });

    setOpen(true);
    expect(await fieldAfterOpen(/^API_KEY/)).toHaveValue("");
  });

  it("reports the empty state instead of a form when nothing is required", async () => {
    renderModal(statusWith([]));

    expect(await screen.findByText("No per-user fields configured for this server.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Credentials" })).not.toBeInTheDocument();
  });

  it("closes and reports the saved status on success", async () => {
    const user = setup();
    const saved = statusWith([{ name: "API_KEY", description: null, is_set: true }]);
    vi.mocked(networking.storeMCPUserEnvVars).mockResolvedValue(saved);
    const { onSaved, onClose } = renderModal(statusWith([{ name: "API_KEY", description: null, is_set: false }]));

    fireEvent.change(await fieldAfterOpen(/^API_KEY/), { target: { value: "abc" } });
    await save(user);

    await waitFor(() => {
      expect(onSaved).toHaveBeenCalledWith(saved);
    });
    expect(onClose).toHaveBeenCalled();
  });

  it("reveals and re-masks the value through the visibility toggle", async () => {
    const user = setup();
    renderModal(statusWith([{ name: "API_KEY", description: null, is_set: false }]));

    fireEvent.change(await fieldAfterOpen(/^API_KEY/), { target: { value: "hunter2" } });
    await user.click(screen.getByRole("button", { name: "Show password" }));
    expect(screen.getByLabelText(/^API_KEY/)).toHaveAttribute("type", "text");
    expect(screen.getByLabelText(/^API_KEY/)).toHaveValue("hunter2");

    await user.click(screen.getByRole("button", { name: "Hide password" }));
    expect(screen.getByLabelText(/^API_KEY/)).toHaveAttribute("type", "password");
  });

  it("does not submit when the visibility toggle is clicked", async () => {
    const user = setup();
    renderModal(statusWith([{ name: "API_KEY", description: null, is_set: true }]));

    await fieldAfterOpen(/^API_KEY/);
    await user.click(screen.getByRole("button", { name: "Show password" }));

    expect(networking.storeMCPUserEnvVars).not.toHaveBeenCalled();
  });

  it("surfaces a save failure without closing", async () => {
    const user = setup();
    vi.mocked(networking.storeMCPUserEnvVars).mockRejectedValue(new Error("boom"));
    const { onSaved, onClose } = renderModal(statusWith([{ name: "API_KEY", description: null, is_set: false }]));

    fireEvent.change(await fieldAfterOpen(/^API_KEY/), { target: { value: "abc" } });
    await save(user);

    await waitFor(() => {
      expect(networking.storeMCPUserEnvVars).toHaveBeenCalledTimes(1);
    });
    expect(onSaved).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });
});
