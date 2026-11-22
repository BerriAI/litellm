import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { ItemSelect } from "./ItemSelect";

it("displays the selected label and reports the value of the option picked", async () => {
  const user = userEvent.setup();
  const onValueChange = vi.fn();
  render(
    <>
      <label htmlFor="client">Client</label>
      <ItemSelect
        id="client"
        value="a"
        items={[
          { value: "a", label: "First client" },
          { value: "b", label: "Second client" },
        ]}
        onValueChange={onValueChange}
      />
    </>,
  );

  const trigger = screen.getByRole("combobox", { name: "Client" });
  expect(trigger).toHaveTextContent("First client");
  await user.click(trigger);
  await user.click(await screen.findByRole("option", { name: "Second client" }));
  expect(onValueChange).toHaveBeenCalledWith("b");
});
