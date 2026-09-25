import userEvent from "@testing-library/user-event";
import { chooseSelectOption, fireEvent, screen } from "./test-utils";

const groups: Record<string, string> = {
  "Classification Method": "Classifier tuning",
  "Heuristic Keyword Overrides": "Classifier tuning",
  "Ignore Custom Tags": "Classifier tuning",
  Affinity: "Sessions and efficiency",
  "Adaptive Routing": "Sessions and efficiency",
  Compression: "Sessions and efficiency",
  "Response Format": "Compatibility",
};

export const openAutoRouterAdvanced = (section: string) => {
  const advanced = screen.getByRole("button", { name: /^Advanced settings/ });
  if (advanced.getAttribute("aria-expanded") !== "true") fireEvent.click(advanced);
  const group = screen.getByRole("button", { name: groups[section] ?? "Routing rules and recovery" });
  if (group.getAttribute("aria-expanded") !== "true") fireEvent.click(group);
};

export const selectAutoRouterOption = async (field: string, option: string) => {
  if (field === "Heuristic" || field === "Routing approach") {
    const user = userEvent.setup();
    fireEvent.click(screen.getByRole("button", { name: field }));
    await user.click(await screen.findByRole("menuitemradio", { name: new RegExp(`^${option}`) }));
    return;
  }
  await chooseSelectOption(userEvent.setup(), screen.getByRole("combobox", { name: field }), new RegExp(`^${option}`));
};

export const selectAutoRouterApproach = async (option: string) => {
  await userEvent.click(screen.getByRole("radio", { name: "LLM" }));
  await selectAutoRouterOption("Routing approach", option);
};
