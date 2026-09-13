import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect } from "vitest";

export async function selectOption(labelText: string, optionText: string) {
  const user = userEvent.setup({ delay: null });
  await user.click(screen.getByLabelText(labelText));

  const option = (await screen.findAllByRole("option")).find((el) => el.textContent?.includes(optionText));
  expect(option).toBeTruthy();
  await user.click(option!);
}
