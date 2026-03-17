import { screen } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import { describe, expect, it } from "vitest";
import ImpactPreviewAlert from "./impact_preview_alert";

const globalImpact = {
  affected_keys_count: -1,
  affected_teams_count: -1,
  sample_keys: [],
  sample_teams: [],
};

const specificImpact = {
  affected_keys_count: 3,
  affected_teams_count: 1,
  sample_keys: ["sk-abc", "sk-def", "sk-ghi"],
  sample_teams: ["team-alpha"],
};

describe("ImpactPreviewAlert", () => {
  it("should render", () => {
    renderWithProviders(<ImpactPreviewAlert impactResult={specificImpact} />);
    expect(screen.getByText("Impact Preview")).toBeInTheDocument();
  });

  describe("when affected_keys_count is -1 (global scope)", () => {
    it("should show a warning about all keys and teams being affected", () => {
      renderWithProviders(<ImpactPreviewAlert impactResult={globalImpact} />);
      expect(screen.getByText(/all keys and teams/i)).toBeInTheDocument();
    });
  });

  describe("when scope is specific", () => {
    it("should show the number of affected keys", () => {
      renderWithProviders(<ImpactPreviewAlert impactResult={specificImpact} />);
      expect(screen.getByText(/3 keys/i)).toBeInTheDocument();
    });

    it("should show the number of affected teams", () => {
      renderWithProviders(<ImpactPreviewAlert impactResult={specificImpact} />);
      expect(screen.getByText(/1 team\b/i)).toBeInTheDocument();
    });

    it("should render sample key tags", () => {
      renderWithProviders(<ImpactPreviewAlert impactResult={specificImpact} />);
      expect(screen.getByText("sk-abc")).toBeInTheDocument();
      expect(screen.getByText("sk-def")).toBeInTheDocument();
    });

    it("should render sample team tags", () => {
      renderWithProviders(<ImpactPreviewAlert impactResult={specificImpact} />);
      expect(screen.getByText("team-alpha")).toBeInTheDocument();
    });

    it("should show a '...more' indicator when there are more than 5 keys", () => {
      const manyKeys = {
        affected_keys_count: 10,
        affected_teams_count: 0,
        sample_keys: ["k1", "k2", "k3", "k4", "k5"],
        sample_teams: [],
      };
      renderWithProviders(<ImpactPreviewAlert impactResult={manyKeys} />);
      expect(screen.getByText(/and 5 more/i)).toBeInTheDocument();
    });

    it("should use singular 'key' when exactly one key is affected", () => {
      const oneKey = { affected_keys_count: 1, affected_teams_count: 0, sample_keys: ["sk-1"], sample_teams: [] };
      renderWithProviders(<ImpactPreviewAlert impactResult={oneKey} />);
      expect(screen.getByText(/1 key\b/i)).toBeInTheDocument();
    });

    it("should not show a key section when there are no sample keys", () => {
      const noKeys = { affected_keys_count: 0, affected_teams_count: 2, sample_keys: [], sample_teams: ["t1", "t2"] };
      renderWithProviders(<ImpactPreviewAlert impactResult={noKeys} />);
      expect(screen.queryByText(/^Keys:/i)).not.toBeInTheDocument();
    });
  });
});
