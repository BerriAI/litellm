---
name: rust-string-enums
description: Define or refactor Rust string-valued enums and their Serde adapters in litellm-rust, using Strum and serde_with while preserving parsing, wire values, and schemas
---

# Rust string enums

Use this skill when adding or changing enums represented by a single string, or surveying handwritten string conversions

## Choose the representation

For an enum with fixed spellings and an unknown-string fallback, prefer `strum::EnumString` and `strum::Display` together with `serde_with::DeserializeFromStr` and `serde_with::SerializeDisplay`. Keep each wire spelling in the Strum attributes instead of repeating it in a handwritten Serde match

Strum implements string conversion traits, not Serde traits. `EnumString` implements `FromStr`; `Display` formats the wire string. The two `serde_with` derives connect those traits to Serde. `AsRefStr` provides a borrowed string accessor and is optional

```rust
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::AsRefStr,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
pub enum EventType {
    #[strum(serialize = "event.created")]
    Created,
    #[strum(default, transparent)]
    Other(String),
}
```

Spell each variant explicitly with `#[strum(serialize = "...")]`; do not use `serialize_all`. With multiple accepted aliases, set `to_string` to the existing canonical output: Strum Display otherwise selects the longest `serialize` spelling

Use only the derives the contract needs. A deserialize-only type should remain deserialize-only. Do not add an unknown variant to a closed enum, or derive Serde for a type that currently has no serialization contract

Plain Serde derives with `rename` or `rename_all` remain appropriate for closed unit enums. Adding Strum and serde_with solely to replace working Serde derives adds little value. When a type has both Serde and Strum parsing, compare their accepted inputs before sharing the parser: case-insensitive Strum parsing must not silently make strict JSON parsing case-insensitive

## Preserve behavior during migration

Read the type, its callers, and existing tests before changing it. Preserve canonical output, accepted aliases, case sensitivity, whitespace handling, unknown values, malformed-input rejection, and existing public conversion contracts. Wrappers that only delegate to a Strum-derived trait are removed rather than kept: callers use `FromStr` and `From<Enum> for &'static str` directly. Keep conversions that add behavior needed by callers or compatibility even when Serde no longer uses them

Use the workspace dependencies and enable `serde_with.workspace = true` in a crate only when needed. Check the versions and enabled features in `Cargo.toml` and `Cargo.lock` rather than upgrading dependencies for this refactor

Remove replaced manual Serde implementations and obsolete Serde conversion attributes. Do not combine the new derives with `wire_type`, `request_type`, or `response_type` aliases that already derive the same Serde traits. Expand the necessary non-Serde derives and schema attributes locally rather than changing shared aliases for unrelated types

Preserve the generated schema, including titles and definition names. Open string enums need a string schema, including unknown values. When replacing Serde `from`/`into` attributes that previously supplied that schema, retain their schema behavior with `#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]` and compare the full generated result. `schemars(with = "String")` also makes a string schema, but changes the schema name and title, so keep it only where it already matches the contract

Keep custom `FromStr` and `Display` implementations for structured strings or validation that Strum does not express faithfully. Their Serde adapters can still use `DeserializeFromStr` and `SerializeDisplay`. Do not replace JSON visitors, tagged payload enums, permissive value wrappers, or domain transformations with string parsing

For a requested survey, document candidates and exceptions without migrating source. If the user asks to approve a bulk migration, present the concrete scope and wait for that approval

## Verify the contract

Extend existing mapped tests with named `rstest` cases. Assert both parsing into the expected variant and serialization to the expected wire string. Include unknown and empty strings for open enums, accepted aliases when present, and rejection of non-string JSON. A decode-only assertion does not prove a round trip

Test structured parsers with valid and invalid payloads, including their existing error behavior. For types with schema support, check the string schema and run the affected crate tests with the schema feature enabled. Run affected downstream checks when conversion APIs or derive aliases change

## Upstream references

The workspace used Strum 0.28.0 and serde_with 3.16.1 when this guidance was written. Consult the matching version of the [EnumString docs](https://docs.rs/strum/0.28.0/strum/derive.EnumString.html), [Display docs](https://docs.rs/strum_macros/0.28.0/strum_macros/derive.Display.html), [DeserializeFromStr docs](https://docs.rs/serde_with/3.16.1/serde_with/derive.DeserializeFromStr.html), and [SerializeDisplay docs](https://docs.rs/serde_with/3.16.1/serde_with/derive.SerializeDisplay.html). Schemars documents [schema overrides and Serde conversion attributes](https://docs.rs/schemars/1.2.2/schemars/derive.JsonSchema.html)
