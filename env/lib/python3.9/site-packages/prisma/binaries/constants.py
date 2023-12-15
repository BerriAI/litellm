from . import platform


__all__ = ('PRISMA_CLI_NAME',)


# local file path for the prisma CLI
if platform.name() == 'windows':
    PRISMA_CLI_NAME = f'prisma-cli-{platform.name()}.exe'
else:
    PRISMA_CLI_NAME = f'prisma-cli-{platform.name()}'  # pyright: ignore[reportConstantRedefinition]
