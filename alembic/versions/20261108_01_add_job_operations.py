"""add job and social publishing operation fields

Revision ID: 20261108_01
Revises:
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from app.db.models import Base

revision: str = "20261108_01"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    if not {"users", "campaigns", "video_jobs", "social_posts"}.issubset(tables):
        Base.metadata.create_all(bind=bind)
        tables = set(inspect(bind).get_table_names())

    video_columns = {column["name"] for column in inspect(bind).get_columns("video_jobs")}
    if "retry_count" not in video_columns:
        op.add_column("video_jobs", sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"))
    if "started_at" not in video_columns:
        op.add_column("video_jobs", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))

    social_columns = {column["name"] for column in inspect(bind).get_columns("social_posts")}
    if "provider_status" not in social_columns:
        op.add_column("social_posts", sa.Column("provider_status", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("social_posts", "provider_status")
    op.drop_column("video_jobs", "started_at")
    op.drop_column("video_jobs", "retry_count")
