import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { TagsInput, type TagsInputOption } from "./TagsInput";

const Harness = ({
  initial = [],
  options,
  onValueChange,
}: {
  initial?: string[];
  options?: TagsInputOption[];
  onValueChange?: (value: string[]) => void;
}) => {
  const [value, setValue] = useState<string[]>(initial);
  return (
    <TagsInput
      value={value}
      options={options}
      tokenSeparators={[","]}
      placeholder="tags"
      onValueChange={(next) => {
        onValueChange?.(next);
        setValue(next);
      }}
    />
  );
};

describe("TagsInput", () => {
  it("commits each token separated value and keeps the unterminated remainder in the field", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness onValueChange={onValueChange} />);

    const input = screen.getByRole("combobox");
    await user.type(input, "acme,globex,initech");

    expect(onValueChange).toHaveBeenLastCalledWith(["acme", "globex"]);
    expect(input).toHaveValue("initech");
  });

  it("commits the pending value when the field loses focus", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness onValueChange={onValueChange} />);

    await user.type(screen.getByRole("combobox"), "acme");
    await user.tab();

    expect(onValueChange).toHaveBeenLastCalledWith(["acme"]);
  });

  it("commits the pending value on Enter without submitting the surrounding form", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn((event: React.FormEvent) => event.preventDefault());
    const onValueChange = vi.fn();
    render(
      <form onSubmit={onSubmit}>
        <Harness onValueChange={onValueChange} />
        <button type="submit">Save</button>
      </form>,
    );

    await user.type(screen.getByRole("combobox"), "acme{Enter}");

    expect(onValueChange).toHaveBeenLastCalledWith(["acme"]);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("ignores a value that is already a tag", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness initial={["acme"]} onValueChange={onValueChange} />);

    await user.type(screen.getByRole("combobox"), "acme,");

    expect(onValueChange).not.toHaveBeenCalled();
  });

  it("labels a chip with the matching option label rather than the raw value", () => {
    render(<Harness initial={["qatar airways"]} options={[{ value: "qatar airways", label: "Qatar Airways (qr)" }]} />);

    expect(screen.getByLabelText("Qatar Airways (qr)")).toBeInTheDocument();
  });
});
