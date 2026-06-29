from typing import Any, Dict, Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from litellm._logging import verbose_router_logger
from litellm.router_strategy.adept_router.store.models import (
    Base,
    Conversation,
    Template,
)
from litellm.router_strategy.adept_router.store.store_template import AdeptTemplateStore


class PostgresTemplateRepo(AdeptTemplateStore):
    """PostgreSQL-backed template/conversation store using SQLAlchemy ORM."""

    def __init__(self, db_url: str) -> None:
        if not db_url:
            raise ValueError(
                "A PostgreSQL connection URL is required. Example: postgresql+psycopg2://user:password@host:5432/dbname"
            )

        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.Session = sessionmaker(bind=self.engine)
        self._initialize_db()

    def _initialize_db(self) -> None:
        try:
            Base.metadata.create_all(self.engine)
            verbose_router_logger.info("AdeptRouter: initialized PostgreSQL template store.")
        except Exception as e:
            verbose_router_logger.error(f"AdeptRouter: error initializing PostgreSQL database: {str(e)}")
            raise

    def match_by_hash(self, template_hash: str, router_id: str) -> Optional[str]:
        try:
            with self.Session() as session:
                row = session.query(Template).filter_by(router_id=router_id, template_hash=template_hash).first()
                return row.id if row else None
        except Exception as e:
            verbose_router_logger.error(f"Error matching template by hash: {str(e)}")
            return None

    def store_conversation(
        self,
        prompt: str,
        response: str,
        template_id: Optional[str] = None,
        additional_information: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not template_id:
            verbose_router_logger.error("template_id is required to store a conversation.")
            return False
        try:
            with self.Session() as session:
                session.add(
                    Conversation(
                        template_id=template_id,
                        prompt=prompt,
                        response=response,
                        additional_information=additional_information,
                    )
                )
                session.commit()
                verbose_router_logger.debug(f"Stored conversation for template {template_id}")
            return True
        except Exception as e:
            verbose_router_logger.error(f"Error storing conversation: {str(e)}")
            return False

    def store_template(
        self,
        template_id: str,
        template: str,
        template_hash: str,
        target_model: str,
        router_id: str,
        additional_information: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Insert a new template row. Returns the surviving template_id (either the one we
        inserted or the one that already existed from a concurrent request).

        Uses ON CONFLICT DO NOTHING on the (router_id, template_hash) unique index so that
        two concurrent requests with the same hash are both safe — the second one silently
        no-ops and we re-fetch the winner's id.
        """
        try:
            with self.Session() as session:
                stmt = (
                    insert(Template)
                    .values(
                        id=template_id,
                        template=template,
                        template_hash=template_hash,
                        target_model=target_model,
                        router_id=router_id,
                        additional_information=additional_information,
                    )
                    .on_conflict_do_nothing(index_elements=["router_id", "template_hash"])
                )
                session.execute(stmt)
                session.commit()
                row = session.query(Template).filter_by(router_id=router_id, template_hash=template_hash).first()
                surviving_id = row.id if row else template_id
                verbose_router_logger.debug(f"AdeptRouter: stored template {surviving_id}")
                return surviving_id
        except Exception as e:
            verbose_router_logger.error(f"AdeptRouter: error storing template: {str(e)}")
            return None

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.Session() as session:
                row = session.execute(select(Template).where(Template.id == template_id)).scalar_one_or_none()
                if row:
                    return {
                        "id": row.id,
                        "template": row.template,
                        "template_hash": row.template_hash,
                        "target_model": row.target_model,
                        "router_id": row.router_id,
                        "additional_information": row.additional_information,
                        "created_at": row.created_at,
                    }
            return None
        except Exception as e:
            verbose_router_logger.error(f"Error retrieving template: {str(e)}")
            return None

    def count_conversation_by_template_id(self, template_id: str) -> Optional[int]:
        try:
            with self.Session() as session:
                count = session.scalar(select(func.count()).where(Conversation.template_id == template_id))
                return count or 0
        except Exception as e:
            verbose_router_logger.error(f"Error counting conversations: {str(e)}")
            return None
