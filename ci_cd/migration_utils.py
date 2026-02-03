"""
Utility functions for processing database migrations.

This module provides functions to make SQL migrations idempotent by adding
IF NOT EXISTS clauses to ADD COLUMN and CREATE INDEX statements.
"""

import re


def make_migration_idempotent(sql_content: str) -> str:
    """
    Post-process SQL migration to make it idempotent by adding IF NOT EXISTS clauses.
    
    This function adds IF NOT EXISTS to:
    - ADD COLUMN statements
    - CREATE INDEX statements
    - CREATE UNIQUE INDEX statements
    
    It safely handles:
    - Already idempotent migrations (won't duplicate IF NOT EXISTS)
    - Multi-line SQL statements
    - Different whitespace patterns
    - Case-insensitive matching
    
    Args:
        sql_content: Raw SQL from Prisma migrate diff
        
    Returns:
        SQL with IF NOT EXISTS clauses added to appropriate statements
        
    Examples:
        >>> sql = 'ALTER TABLE "Test" ADD COLUMN "col1" TEXT;'
        >>> make_migration_idempotent(sql)
        'ALTER TABLE "Test" ADD COLUMN IF NOT EXISTS "col1" TEXT;'
        
        >>> sql = 'CREATE INDEX "idx1" ON "Test"("col1");'
        >>> make_migration_idempotent(sql)
        'CREATE INDEX IF NOT EXISTS "idx1" ON "Test"("col1");'
        
        >>> sql = 'ALTER TABLE "Test" ADD COLUMN IF NOT EXISTS "col1" TEXT;'
        >>> make_migration_idempotent(sql)
        'ALTER TABLE "Test" ADD COLUMN IF NOT EXISTS "col1" TEXT;'
    """
    if not sql_content or not sql_content.strip():
        return sql_content
    
    # Add IF NOT EXISTS to ADD COLUMN statements (only if not already present)
    # Pattern matches: ADD COLUMN followed by whitespace and a quoted identifier
    # Uses negative lookahead to avoid matching if IF NOT EXISTS is already present
    sql_content = re.sub(
        r'ADD COLUMN\s+(?!IF NOT EXISTS\s+)("[\w]+")',
        r'ADD COLUMN IF NOT EXISTS \1',
        sql_content,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # Add IF NOT EXISTS to CREATE INDEX statements (only if not already present)
    sql_content = re.sub(
        r'CREATE INDEX\s+(?!IF NOT EXISTS\s+)("[\w]+")',
        r'CREATE INDEX IF NOT EXISTS \1',
        sql_content,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # Add IF NOT EXISTS to CREATE UNIQUE INDEX statements (only if not already present)
    # Must come after CREATE INDEX to avoid partial matches
    sql_content = re.sub(
        r'CREATE UNIQUE INDEX\s+(?!IF NOT EXISTS\s+)("[\w]+")',
        r'CREATE UNIQUE INDEX IF NOT EXISTS \1',
        sql_content,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    return sql_content
