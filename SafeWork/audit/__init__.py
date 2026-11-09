"""Módulo 3: evidências, metadados e trilha de auditoria."""

from safework.audit.auditor import ViolationAuditor, ViolationNotifier
from safework.audit.evidence import EvidenceStore
from safework.audit.records import EmailStatus, ViolationRecord
from safework.audit.repository import AuditRepository

__all__ = [
    "AuditRepository",
    "EmailStatus",
    "EvidenceStore",
    "ViolationAuditor",
    "ViolationNotifier",
    "ViolationRecord",
]
