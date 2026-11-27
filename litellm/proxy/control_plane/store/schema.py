from typing import Final

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, MetaData, Table, Text, UniqueConstraint, false, func
from sqlalchemy.dialects.postgresql import JSONB

metadata: Final = MetaData()
dataplanes: Final = Table(
    "control_plane_dataplanes",
    metadata,
    Column("id", Text, primary_key=True),
    Column("spec", JSONB, nullable=False),
    Column("revision", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_by", Text, nullable=False),
)
roles: Final = Table(
    "control_plane_roles",
    metadata,
    Column("name", Text, primary_key=True),
    Column("scopes", JSONB, nullable=False),
    Column("inherits", JSONB, nullable=False),
    Column("revision", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_by", Text, nullable=False),
)
fleet_teams: Final = Table(
    "control_plane_fleet_teams",
    metadata,
    Column("id", Text, primary_key=True),
    Column("team_alias", Text, nullable=False),
    Column("desired", JSONB(none_as_null=True)),
    Column("revision", Text, nullable=False),
    Column("updated_by", Text, nullable=False),
)
fleet_team_dataplanes: Final = Table(
    "control_plane_fleet_team_dataplanes",
    metadata,
    Column("fleet_team_id", Text, ForeignKey("control_plane_fleet_teams.id"), primary_key=True),
    Column("dataplane_id", Text, primary_key=True),
    Column("dataplane_url", Text, nullable=False),
    Column("team_id", Text, nullable=False),
    Column("applied_revision", Text),
    Column("attempted_revision", Text),
    Column("last_error", Text),
    Column("unreachable", Boolean, nullable=False, server_default=false()),
    UniqueConstraint("dataplane_id", "team_id"),
)
