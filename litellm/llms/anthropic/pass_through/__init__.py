from litellm.litellm_core_utils.renamed_package_shim import alias_renamed_package

__getattr__ = alias_renamed_package(
    __name__,
    "litellm.llms.anthropic.messages",
    nested_renames=((f"{__name__}.messages", "litellm.llms.anthropic.messages.native"),),
)
