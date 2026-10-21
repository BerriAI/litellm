import userEvent from "@testing-library/user-event";
import { chooseSelectOption, fireEvent, screen } from "./test-utils";

const groups: Record<string, string> = {
  "Heuristic Keyword Overrides": "Heuristic tuning",
  "Ignore Custom Tags": "Request preprocessing",
  Affinity: "Sessions and efficiency",
  "Adaptive Routing": "Sessions and efficiency",
  "Cache-aware routing": "Sessions and efficiency",
  Compression: "Sessions and efficiency",
  "Response Format": "Compatibility",
};

export const openAutoRouterAdvanced = (section: string) => {
  const advanced = screen.getByRole("button", { name: /^Advanced settings/ });
  if (advanced.getAttribute("aria-expanded") !== "true") fireEvent.click(advanced);
  const labels =
    section === "Classification Method"
      ? ["Heuristic tuning", "LLM tuning", "Classifier tuning"]
      : [groups[section] ?? "Routing rules and recovery"];
  for (const label of labels) {
    const group = screen.queryByRole("button", { name: label });
    if (group && group.getAttribute("aria-expanded") !== "true") fireEvent.click(group);
  }
};

export const selectAutoRouterOption = async (field: string, option: string) => {
  if (field === "Heuristic" || field === "Routing approach" || field === "Heuristic before the judge") {
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
