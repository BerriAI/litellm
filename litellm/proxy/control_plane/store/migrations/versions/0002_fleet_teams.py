from typing import Final

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: Final = "0002"
down_revision: Final = "0001"
branch_labels: Final = None
depends_on: Final = None


def upgrade() -> None:
    op.create_table(
        "control_plane_fleet_teams",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("team_alias", sa.Text(), nullable=False),
        sa.Column("desired", postgresql.JSONB(none_as_null=True)),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("updated_by", sa.Text(), nullable=False),
    )
    op.create_table(
        "control_plane_fleet_team_dataplanes",
        sa.Column("fleet_team_id", sa.Text(), sa.ForeignKey("control_plane_fleet_teams.id"), primary_key=True),
        sa.Column("dataplane_id", sa.Text(), primary_key=True),
        sa.Column("dataplane_url", sa.Text(), nullable=False),
        sa.Column("team_id", sa.Text(), nullable=False),
        sa.Column("applied_revision", sa.Text()),
        sa.Column("attempted_revision", sa.Text()),
        sa.Column("last_error", sa.Text()),
        sa.Column("unreachable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("dataplane_id", "team_id"),
    )


def downgrade() -> None:
    op.drop_table("control_plane_fleet_team_dataplanes")
    op.drop_table("control_plane_fleet_teams")
