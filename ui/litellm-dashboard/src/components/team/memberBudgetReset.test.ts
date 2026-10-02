import { describe, expect, it } from "vitest";
import { isMemberBudgetChanging, memberBudgetUpdateMessage } from "./memberBudgetReset";

describe("member budget updates", () => {
  it.each([20, "20"])("offers choices for a changed positive amount %s", (amount) => {
    expect(isMemberBudgetChanging(amount, 10)).toBe(true);
    expect(isMemberBudgetChanging(amount, null)).toBe(true);
  });

  it.each([10, "10", 0, "0", "", null, undefined, -5, "invalid", Infinity])(
    "does not apply a bulk action for unchanged or non-positive amount %s",
    (amount) => {
      expect(isMemberBudgetChanging(amount, 10)).toBe(false);
    },
  );

  it("reports the committed member count", () => {
    expect(memberBudgetUpdateMessage(1)).toBe(
      "Team settings updated. 1 member budget now follows the team default amount",
    );
    expect(memberBudgetUpdateMessage(4)).toBe(
      "Team settings updated. 4 member budgets now follow the team default amount",
    );
  });
});
