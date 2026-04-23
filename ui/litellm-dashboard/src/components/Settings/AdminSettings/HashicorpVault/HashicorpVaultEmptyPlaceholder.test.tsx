import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import HashicorpVaultEmptyPlaceholder from "./HashicorpVaultEmptyPlaceholder";

describe("HashicorpVaultEmptyPlaceholder", () => {
  it("should render the empty state message and configure button", () => {
    render(<HashicorpVaultEmptyPlaceholder onAdd={vi.fn()} />);
    expect(screen.getByText("No Vault Configuration Found")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /configure vault/i })).toBeInTheDocument();
  });

  it("should call onAdd when the configure button is clicked", async () => {
    const onAdd = vi.fn();
    const user = userEvent.setup();
    render(<HashicorpVaultEmptyPlaceholder onAdd={onAdd} />);

    await user.click(screen.getByRole("button", { name: /configure vault/i }));

    expect(onAdd).toHaveBeenCalledOnce();
  });

  it("should display the description text about Vault purpose", () => {
    render(<HashicorpVaultEmptyPlaceholder onAdd={vi.fn()} />);
    expect(
      screen.getByText(/Configure Hashicorp Vault to securely manage provider API keys/),
    ).toBeInTheDocument();
  });
});
