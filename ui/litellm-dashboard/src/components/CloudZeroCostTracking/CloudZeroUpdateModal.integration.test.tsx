import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CloudZeroUpdateModal from "./CloudZeroUpdateModal";
import { CloudZeroSettings } from "./types";

const mutate = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => ({ accessToken: "test-token" }),
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings", () => ({
  useCloudZeroUpdateSettings: () => ({ mutate, isPending: false }),
}));

const STORED_SETTINGS: CloudZeroSettings = {
  connection_id: "stored-connection-id",
  api_key_masked: "sk-cz-****last4",
  timezone: "Europe/Berlin",
  status: "Active",
};

const renderModal = (settings: CloudZeroSettings = STORED_SETTINGS) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <CloudZeroUpdateModal open={true} onOk={vi.fn()} onCancel={vi.fn()} settings={settings} />
    </QueryClientProvider>,
  );
};

const submittedPayload = (): Record<string, unknown> => {
  expect(mutate).toHaveBeenCalledTimes(1);
  return mutate.mock.calls[0][0] as Record<string, unknown>;
};

describe("CloudZeroUpdateModal submit payload", () => {
  beforeEach(() => {
    mutate.mockClear();
  });

  it("seeds the stored connection id and timezone but never the stored key", () => {
    renderModal();

    expect(screen.getByLabelText("Connection ID")).toHaveValue("stored-connection-id");
    expect(screen.getByLabelText("Timezone")).toHaveValue("Europe/Berlin");
    expect(screen.getByLabelText("CloudZero API Key")).toHaveValue("");
  });

  it("omits api_key entirely when the key field is left untouched, preserving the stored secret", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole("button", { name: "Update" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "stored-connection-id",
        timezone: "Europe/Berlin",
      }),
    );
    expect("api_key" in submittedPayload()).toBe(false);
  });

  it("sends api_key only once the user types a replacement key", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "cz-rotated-key" } });
    await user.click(screen.getByRole("button", { name: "Update" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "stored-connection-id",
        timezone: "Europe/Berlin",
        api_key: "cz-rotated-key",
      }),
    );
  });

  it("falls back to UTC when the stored timezone is cleared", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText("Timezone"));
    await user.click(screen.getByRole("button", { name: "Update" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "stored-connection-id",
        timezone: "UTC",
      }),
    );
  });

  it("falls back to UTC when the stored settings carry no timezone", () => {
    renderModal({ ...STORED_SETTINGS, timezone: null });

    expect(screen.getByLabelText("Timezone")).toHaveValue("UTC");
  });

  it("reports a null connection id from the server as the required field, not as a type error", async () => {
    const user = userEvent.setup();
    renderModal({ ...STORED_SETTINGS, connection_id: null, timezone: null });

    await user.click(screen.getByRole("button", { name: "Update" }));

    expect(await screen.findByText("Please enter your CloudZero connection ID")).toBeInTheDocument();
    expect(screen.queryByText(/expected string, received null/i)).not.toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("blocks submission when the connection id is cleared, and leaves the key optional", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText("Connection ID"));
    await user.click(screen.getByRole("button", { name: "Update" }));

    expect(await screen.findByText("Please enter your CloudZero connection ID")).toBeInTheDocument();
    expect(screen.queryByText("Please enter your CloudZero API key")).not.toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("does not submit when Enter is pressed inside a text field", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByLabelText("CloudZero API Key"), "cz-rotated-key{Enter}");

    expect(mutate).not.toHaveBeenCalled();
  });
});
