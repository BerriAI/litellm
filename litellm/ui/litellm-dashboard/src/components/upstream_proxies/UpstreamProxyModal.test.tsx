import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { UpstreamProxy } from "./api";
import UpstreamProxyModal from "./UpstreamProxyModal";

const STORED: UpstreamProxy = {
  name: "jp",
  protocol: "http",
  host: "10.80.0.64",
  port: 3128,
  username: "probe",
  has_password: true,
  description: "Tokyo exit",
  used_by_models: [],
  used_by_credentials: [],
};

const fill = async (label: RegExp, value: string) => {
  const input = screen.getByLabelText(label);
  await userEvent.clear(input);
  if (value) {
    await userEvent.type(input, value);
  }
};

describe("UpstreamProxyModal", () => {
  it("creates an HTTP proxy on the default port and leaves unset optional fields out of the request", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<UpstreamProxyModal mode="add" open onCancel={vi.fn()} onSubmit={onSubmit} />);

    await fill(/^Name/, "jp-tokyo");
    await fill(/^Host/, "10.80.0.64");
    await userEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const expected = { name: "jp-tokyo", protocol: "http", host: "10.80.0.64", port: 8080 };
    expect(onSubmit).toHaveBeenCalledWith(expected);
  });

  it.each([
    ["a host with a scheme", /^Host/, "http://10.80.0.64", /without a scheme or path/],
    ["a port above 65535", /^Port/, "70000", /between 1 and 65535/],
    ["a name with spaces", /^Name/, "jp tokyo", /at most 64 characters/],
  ])("refuses %s before calling the API", async (_case, label, value, message) => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<UpstreamProxyModal mode="add" open onCancel={vi.fn()} onSubmit={onSubmit} />);
    await fill(/^Name/, "jp");
    await fill(/^Host/, "10.80.0.64");

    await fill(label, value);
    await userEvent.click(screen.getByRole("button", { name: "Create" }));

    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("keeps the saved password when editing other fields and does not let the name change", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<UpstreamProxyModal mode="edit" open proxy={STORED} onCancel={vi.fn()} onSubmit={onSubmit} />);

    expect(screen.getByLabelText(/^Name/)).toBeDisabled();
    await fill(/^Port/, "8888");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const expected = { protocol: "http", host: "10.80.0.64", port: 8888, username: "probe", description: "Tokyo exit" };
    expect(onSubmit).toHaveBeenCalledWith("jp", expected);
  });

  it("sends a newly typed password and an emptied username so the stored ones are replaced", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<UpstreamProxyModal mode="edit" open proxy={STORED} onCancel={vi.fn()} onSubmit={onSubmit} />);

    await fill(/^Username/, "");
    await fill(/^Password/, "n3w");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][1]).toMatchObject({ username: "", password: "n3w" });
  });
});
