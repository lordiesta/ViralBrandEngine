"""add product KPI models

Revision ID: 20261108_02
Revises: 20261108_01
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20261108_02"
down_revision: Union[str, Sequence[str], None] = "20261108_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())

    tables_to_create = {
        "product_events": (
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("event_name", sa.String(128), nullable=False),
            sa.Column("user_id", sa.Uuid(), nullable=True),
            sa.Column("campaign_id", sa.Uuid(), nullable=True),
            sa.Column("job_id", sa.Uuid(), nullable=True),
            sa.Column("properties", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        ),
        "customer_feedback": (
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("feedback_type", sa.String(16), nullable=False),
            sa.Column("score", sa.Integer(), nullable=False),
            sa.Column("comment", sa.Text(), nullable=True),
            sa.Column("user_id", sa.Uuid(), nullable=True),
            sa.Column("job_id", sa.Uuid(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        ),
        "content_costs": (
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("job_id", sa.Uuid(), nullable=False),
            sa.Column("provider", sa.String(128), nullable=False),
            sa.Column("amount", sa.Float(), nullable=False),
            sa.Column("currency", sa.String(8), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        ),
        "social_performance": (
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("social_post_id", sa.Uuid(), nullable=False),
            sa.Column("views", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("likes", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("comments", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("shares", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("saves", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("measured_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        ),
    }
    for table_name, columns in tables_to_create.items():
        if table_name not in tables:
            op.create_table(table_name, *columns)


def downgrade() -> None:
    for table_name in ("social_performance", "content_costs", "customer_feedback", "product_events"):
        op.drop_table(table_name)
