from typing import Final

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: Final = "0001"
down_revision: Final = None
branch_labels: Final = None
depends_on: Final = None


def upgrade() -> None:
    op.create_table(
        "control_plane_dataplanes",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("spec", postgresql.JSONB(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.Text(), nullable=False),
    )
    op.create_table(
        "control_plane_roles",
        sa.Column("name", sa.Text(), primary_key=True),
        sa.Column("scopes", postgresql.JSONB(), nullable=False),
        sa.Column("inherits", postgresql.JSONB(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("control_plane_roles")
    op.drop_table("control_plane_dataplanes")
