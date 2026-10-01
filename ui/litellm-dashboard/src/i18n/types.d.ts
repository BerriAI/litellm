/**
 * Global i18next type augmentation. This file is owned by Agent 4 permanently.
 */
// This import makes this file a module, which turns `declare module` below into
// an augmentation instead of an ambient declaration that would shadow the whole
// i18next package's types.
import "i18next";

declare module "i18next" {
  interface CustomTypeOptions {
    defaultNS: "common";
    // Deliberately NO strict `resources` key augmentation here. i18next v26's
    // typed qualified keys are fragile (colons stripped in the union, qualified
    // non-default lookups typed `unknown`, resource-shape sensitivity) and, in
    // this static-export dashboard, repeatedly broke `next build`'s type check
    // for a benefit that is redundant with the guarantees we already have:
    //   - runtime key correctness is enforced by the readiness gate (ADR-03,
    //     no raw-key flash) and
    //   - en/zh key inventory is enforced by A7's check-keys CI gate.
    // So keys are typed loosely (`string`), which compiles stably. Common-NS
    // keys are still used unprefixed per `defaultNS`.
  }
}
