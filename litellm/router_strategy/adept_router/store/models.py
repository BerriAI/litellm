from sqlalchemy import JSON, TIMESTAMP, Column, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Template(Base):
    __tablename__ = "templates"

    id = Column(String, primary_key=True)
    template = Column(String, nullable=False)
    template_hash = Column(String(64), nullable=True)
    router_id = Column(String, nullable=False)
    target_model = Column(String)
    additional_information = Column(JSON)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    conversations = relationship("Conversation", back_populates="template")

    __table_args__ = (Index("ix_templates_router_hash", "router_id", "template_hash", unique=True),)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(String, ForeignKey("templates.id"), nullable=False)
    prompt = Column(String, nullable=False)
    response = Column(String, nullable=False)
    additional_information = Column(JSON)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    template = relationship("Template", back_populates="conversations")

    __table_args__ = (Index("ix_conversations_template_id", "template_id"),)
