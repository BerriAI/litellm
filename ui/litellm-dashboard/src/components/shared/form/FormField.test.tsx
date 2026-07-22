import { zodResolver } from "@hookform/resolvers/zod";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as React from "react";
import { useForm } from "react-hook-form";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod/v4";

import { Input } from "@/components/ui/input";

import { FormField } from "./FormField";

const schema = z.object({
  team_alias: z.string().min(1, "Please input a team name"),
  owner: z.string(),
});

type FormInput = z.input<typeof schema>;

const TestForm = ({
  onSubmit,
  defaultValues = { team_alias: "team-a", owner: "" },
  description,
}: {
  onSubmit: (values: z.output<typeof schema>) => void;
  defaultValues?: FormInput;
  description?: React.ReactNode;
}) => {
  const form = useForm<FormInput, unknown, z.output<typeof schema>>({
    resolver: zodResolver(schema),
    defaultValues,
  });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <FormField control={form.control} name="team_alias" label="Team Name" description={description}>
        {(field) => <Input {...field} />}
      </FormField>
      <button type="submit">Save</button>
    </form>
  );
};

describe("FormField", () => {
  it("associates the label with the control so it is reachable by its accessible name", () => {
    render(<TestForm onSubmit={vi.fn()} />);

    expect(screen.getByLabelText("Team Name")).toHaveValue("team-a");
  });

  it("gives each field instance a unique control id", () => {
    const Harness = () => {
      const form = useForm<FormInput>({ defaultValues: { team_alias: "", owner: "" } });
      return (
        <>
          <FormField control={form.control} name="team_alias" label="One">
            {(field) => <Input {...field} />}
          </FormField>
          <FormField control={form.control} name="owner" label="Two">
            {(field) => <Input {...field} />}
          </FormField>
        </>
      );
    };
    render(<Harness />);

    expect(screen.getByLabelText("One").id).not.toBe(screen.getByLabelText("Two").id);
  });

  it("feeds edits back into form state and submits the parsed output", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<TestForm onSubmit={onSubmit} />);

    await user.clear(screen.getByLabelText("Team Name"));
    await user.type(screen.getByLabelText("Team Name"), "team-b");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toEqual({ team_alias: "team-b", owner: "" });
  });

  it("renders the zod message and blocks submit when validation fails", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<TestForm onSubmit={onSubmit} />);

    await user.clear(screen.getByLabelText("Team Name"));
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Please input a team name");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("marks the control invalid and points aria-describedby at the message", async () => {
    const user = userEvent.setup();
    render(<TestForm onSubmit={vi.fn()} />);

    await user.clear(screen.getByLabelText("Team Name"));
    await user.click(screen.getByRole("button", { name: "Save" }));

    const control = await screen.findByLabelText("Team Name");
    await waitFor(() => expect(control).toHaveAttribute("aria-invalid", "true"));
    expect(control.getAttribute("aria-describedby")).toBe(screen.getByRole("alert").id);
  });

  it("leaves a valid control free of aria-invalid", () => {
    render(<TestForm onSubmit={vi.fn()} />);

    expect(screen.getByLabelText("Team Name")).not.toHaveAttribute("aria-invalid");
  });

  it("clears the message once the value becomes valid again", async () => {
    const user = userEvent.setup();
    render(<TestForm onSubmit={vi.fn()} />);

    await user.clear(screen.getByLabelText("Team Name"));
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Team Name"), "team-c");

    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("describes the control by its description when there is no error", () => {
    render(<TestForm onSubmit={vi.fn()} description="Shown to team members" />);

    const control = screen.getByLabelText("Team Name");
    const describedBy = control.getAttribute("aria-describedby");

    expect(describedBy).not.toBeNull();
    expect(document.getElementById(describedBy!)).toHaveTextContent("Shown to team members");
  });

  it("describes the control by both description and error while invalid", async () => {
    const user = userEvent.setup();
    render(<TestForm onSubmit={vi.fn()} description="Shown to team members" />);

    await user.clear(screen.getByLabelText("Team Name"));
    await user.click(screen.getByRole("button", { name: "Save" }));
    await screen.findByRole("alert");

    const ids = screen.getByLabelText("Team Name").getAttribute("aria-describedby")?.split(" ") ?? [];

    expect(ids).toHaveLength(2);
    expect(ids).toContain(screen.getByRole("alert").id);
  });

  it("omits aria-describedby entirely when there is no description and no error", () => {
    render(<TestForm onSubmit={vi.fn()} />);

    expect(screen.getByLabelText("Team Name")).not.toHaveAttribute("aria-describedby");
  });

  it("hands the control a value and onChange so non-native widgets can be wired", async () => {
    const user = userEvent.setup();
    const seen: unknown[] = [];
    const Harness = () => {
      const form = useForm<FormInput>({ defaultValues: { team_alias: "team-a", owner: "" } });
      return (
        <FormField control={form.control} name="team_alias" label="Team Name">
          {(field) => {
            seen.push(field.value);
            return (
              <button type="button" onClick={() => field.onChange("from-widget")}>
                widget
              </button>
            );
          }}
        </FormField>
      );
    };
    render(<Harness />);

    await user.click(screen.getByRole("button", { name: "widget" }));

    await waitFor(() => expect(seen.at(-1)).toBe("from-widget"));
  });
});
