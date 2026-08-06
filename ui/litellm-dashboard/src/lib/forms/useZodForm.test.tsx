import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod/v4";

import { useZodForm } from "./useZodForm";

const schema = z.object({
  alias: z.string().min(1, "Alias is required"),
  budget: z.string().transform((value) => Number(value)),
});

const TestForm = ({ onSubmit }: { onSubmit: (values: z.output<typeof schema>) => void }) => {
  const form = useZodForm(schema, { defaultValues: { alias: "", budget: "42" } });
  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <input aria-label="Alias" {...form.register("alias")} />
      <button type="submit">Save</button>
    </form>
  );
};

describe("useZodForm", () => {
  it("blocks submit when the schema rejects the values", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<TestForm onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(onSubmit).not.toHaveBeenCalled());
  });

  it("hands submit the transformed output, not the raw input", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<TestForm onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText("Alias"), "acme");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toEqual({ alias: "acme", budget: 42 });
  });
});
