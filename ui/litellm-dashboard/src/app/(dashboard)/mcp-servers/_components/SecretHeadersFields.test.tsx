import { act, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import SecretHeadersFields from "./SecretHeadersFields";
import { McpFormHarness as WithForm } from "./McpFormTestHarness";

const NAME_PLACEHOLDER = "Header name (e.g., DD-API-KEY)";
const VALUE_PLACEHOLDER = "Header value";

describe("SecretHeadersFields", () => {
  it("starts a new server with one empty row whose value is masked", () => {
    render(
      <WithForm>
        <SecretHeadersFields />
      </WithForm>,
    );
    expect(screen.getAllByPlaceholderText(NAME_PLACEHOLDER)).toHaveLength(1);
    expect(screen.getByPlaceholderText(VALUE_PLACEHOLDER)).toHaveAttribute("type", "password");
  });

  it("adds a single starter row even when React runs effects twice", () => {
    render(
      <React.StrictMode>
        <WithForm>
          <SecretHeadersFields />
        </WithForm>
      </React.StrictMode>,
    );
    expect(screen.getAllByPlaceholderText(NAME_PLACEHOLDER)).toHaveLength(1);
  });

  it("starts an existing server with no rows, since the stored headers are never returned", () => {
    render(
      <WithForm>
        <SecretHeadersFields isEditing />
      </WithForm>,
    );
    expect(screen.queryByPlaceholderText(NAME_PLACEHOLDER)).not.toBeInTheDocument();
    expect(screen.getByText(/leave this empty to keep it/)).toBeInTheDocument();
  });

  it("submits every row under secret_headers and drops a removed row", async () => {
    const onFinish = vi.fn();
    render(
      <WithForm onFinish={onFinish}>
        <SecretHeadersFields isEditing />
      </WithForm>,
    );
    const addHeader = screen.getByRole("button", { name: /Add Header/ });
    await act(async () => fireEvent.click(addHeader));
    await act(async () => fireEvent.click(addHeader));
    const names = screen.getAllByPlaceholderText(NAME_PLACEHOLDER);
    const values = screen.getAllByPlaceholderText(VALUE_PLACEHOLDER);
    await act(async () => {
      fireEvent.change(names[0], { target: { value: "DD-API-KEY" } });
      fireEvent.change(values[0], { target: { value: "api-secret" } });
      fireEvent.change(names[1], { target: { value: "DD-APPLICATION-KEY" } });
      fireEvent.change(values[1], { target: { value: "app-secret" } });
    });
    await act(async () => fireEvent.click(screen.getAllByLabelText("Remove header")[1]));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Submit" })));
    expect(onFinish).toHaveBeenCalledWith({ secret_headers: [{ header: "DD-API-KEY", value: "api-secret" }] });
  });

  it("offers removal as a labeled button and keeps the last row of a new server", async () => {
    render(
      <WithForm>
        <SecretHeadersFields />
      </WithForm>,
    );
    const remove = screen.getByRole("button", { name: "Remove header" });
    expect(remove).toBeDisabled();
    await act(async () => fireEvent.click(screen.getByRole("button", { name: /Add Header/ })));
    const buttons = screen.getAllByRole("button", { name: "Remove header" });
    expect(buttons).toHaveLength(2);
    buttons.forEach((button) => expect(button).toBeEnabled());
    await act(async () => fireEvent.click(buttons[1]));
    expect(screen.getAllByPlaceholderText(NAME_PLACEHOLDER)).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Remove header" })).toBeDisabled();
  });

  it("lets a saved server drop every row, since no rows keeps the stored headers", async () => {
    render(
      <WithForm>
        <SecretHeadersFields isEditing />
      </WithForm>,
    );
    await act(async () => fireEvent.click(screen.getByRole("button", { name: /Add Header/ })));
    const remove = screen.getByRole("button", { name: "Remove header" });
    expect(remove).toBeEnabled();
    await act(async () => fireEvent.click(remove));
    expect(screen.queryByPlaceholderText(NAME_PLACEHOLDER)).not.toBeInTheDocument();
  });
});
