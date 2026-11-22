from typing import Final

from sqlalchemy import Column, DateTime, MetaData, Table, Text, func
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
