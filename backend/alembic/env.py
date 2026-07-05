from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Import all models so Alembic can detect them
from app.database import Base
from app.models import (
    RawReview, ClassifiedReview, Theme, ReviewThemeMapping,
    SynthesisCache, QueryLog, QuoteCollection, CollectionItem, PipelineRun
)

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata

import os
from dotenv import load_dotenv

# Load .env file from the root directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '.env'))

# Override the alembic.ini URL with the one from our environment
db_url = os.environ.get("DATABASE_URL_SYNC")
if db_url:
    # Escape % for ConfigParser
    db_url = db_url.replace("%", "%%")
    config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
