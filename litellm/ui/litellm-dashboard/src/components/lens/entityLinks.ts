import { createSerializer, parseAsString, parseAsStringLiteral } from "nuqs/server";
import { uiHref } from "@/utils/uiHref";

export type LensEntityScope =
  | { readonly kind: "team"; readonly teamId: string }
  | { readonly kind: "key"; readonly teamId: string | null; readonly keyHash: string | null | undefined };

const serializeLensEntity = createSerializer({
  tab: parseAsStringLiteral(["traces"]),
  team_id: parseAsString,
  api_key_hash: parseAsString,
});

export function lensEntityHref(scope: LensEntityScope): string | null {
  if (scope.kind === "key" && !/^[a-f\d]{64}$/i.test(scope.keyHash ?? "")) return null;

  return serializeLensEntity(uiHref("lens"), {
    tab: "traces",
    team_id: scope.teamId,
    api_key_hash: scope.kind === "key" ? scope.keyHash?.toLowerCase() : null,
  });
}
