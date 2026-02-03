"""
Tests for migration utility functions.

Tests cover edge cases including:
- Already idempotent migrations
- Multi-line statements
- Different whitespace patterns
- Case variations
- Comments and other SQL statements
- Empty and edge case inputs
"""

import pytest

# Import from ci_cd.migration_utils
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ci_cd.migration_utils import make_migration_idempotent


class TestMakeMigrationIdempotent:
    """Test suite for make_migration_idempotent function."""

    def test_add_column_single_line(self):
        """Test ADD COLUMN in single line format."""
        sql = 'ALTER TABLE "Test" ADD COLUMN     "col1" TEXT;'
        expected = 'ALTER TABLE "Test" ADD COLUMN IF NOT EXISTS "col1" TEXT;'
        assert make_migration_idempotent(sql) == expected

    def test_add_column_multi_line(self):
        """Test ADD COLUMN in multi-line format."""
        sql = '''ALTER TABLE "Test"
    ADD COLUMN     "col1" TEXT,
    ADD COLUMN     "col2" TEXT;'''
        expected = '''ALTER TABLE "Test"
    ADD COLUMN IF NOT EXISTS "col1" TEXT,
    ADD COLUMN IF NOT EXISTS "col2" TEXT;'''
        assert make_migration_idempotent(sql) == expected

    def test_add_column_already_idempotent(self):
        """Test that already idempotent ADD COLUMN is not modified."""
        sql = 'ALTER TABLE "Test" ADD COLUMN IF NOT EXISTS "col1" TEXT;'
        expected = sql  # Should remain unchanged
        assert make_migration_idempotent(sql) == expected

    def test_create_index(self):
        """Test CREATE INDEX statement."""
        sql = 'CREATE INDEX "idx1" ON "Test"("col1");'
        expected = 'CREATE INDEX IF NOT EXISTS "idx1" ON "Test"("col1");'
        assert make_migration_idempotent(sql) == expected

    def test_create_unique_index(self):
        """Test CREATE UNIQUE INDEX statement."""
        sql = 'CREATE UNIQUE INDEX "idx1" ON "Test"("col1");'
        expected = 'CREATE UNIQUE INDEX IF NOT EXISTS "idx1" ON "Test"("col1");'
        assert make_migration_idempotent(sql) == expected

    def test_create_index_already_idempotent(self):
        """Test that already idempotent CREATE INDEX is not modified."""
        sql = 'CREATE INDEX IF NOT EXISTS "idx1" ON "Test"("col1");'
        expected = sql  # Should remain unchanged
        assert make_migration_idempotent(sql) == expected

    def test_create_unique_index_already_idempotent(self):
        """Test that already idempotent CREATE UNIQUE INDEX is not modified."""
        sql = 'CREATE UNIQUE INDEX IF NOT EXISTS "idx1" ON "Test"("col1");'
        expected = sql  # Should remain unchanged
        assert make_migration_idempotent(sql) == expected

    def test_case_insensitive(self):
        """Test that function is case-insensitive."""
        sql = 'alter table "Test" add column     "col1" text;'
        result = make_migration_idempotent(sql)
        assert 'IF NOT EXISTS' in result.upper() or 'if not exists' in result.lower()

    def test_multiple_statements(self):
        """Test multiple statements in one SQL block."""
        sql = '''ALTER TABLE "Test" ADD COLUMN     "col1" TEXT;
CREATE INDEX "idx1" ON "Test"("col1");
CREATE UNIQUE INDEX "idx2" ON "Test"("col2");'''
        result = make_migration_idempotent(sql)
        assert 'ADD COLUMN IF NOT EXISTS' in result
        assert 'CREATE INDEX IF NOT EXISTS' in result
        assert 'CREATE UNIQUE INDEX IF NOT EXISTS' in result

    def test_with_comments(self):
        """Test SQL with comments."""
        sql = '''-- AlterTable
ALTER TABLE "Test" ADD COLUMN     "col1" TEXT;

-- CreateIndex
CREATE INDEX "idx1" ON "Test"("col1");'''
        result = make_migration_idempotent(sql)
        assert '-- AlterTable' in result
        assert 'ADD COLUMN IF NOT EXISTS' in result
        assert 'CREATE INDEX IF NOT EXISTS' in result

    def test_mixed_idempotent_and_non_idempotent(self):
        """Test mix of already idempotent and non-idempotent statements."""
        sql = '''ALTER TABLE "Test" ADD COLUMN IF NOT EXISTS "col1" TEXT;
ALTER TABLE "Test" ADD COLUMN     "col2" TEXT;
CREATE INDEX IF NOT EXISTS "idx1" ON "Test"("col1");
CREATE INDEX "idx2" ON "Test"("col2");'''
        result = make_migration_idempotent(sql)
        # Count occurrences to ensure no duplication
        assert result.count('ADD COLUMN IF NOT EXISTS') == 2
        assert result.count('CREATE INDEX IF NOT EXISTS') == 2
        assert 'ADD COLUMN     "col2"' not in result  # Should be replaced

    def test_complex_index_name(self):
        """Test index names with underscores and numbers."""
        sql = 'CREATE INDEX "LiteLLM_Table_col1_idx_123" ON "Test"("col1");'
        expected = 'CREATE INDEX IF NOT EXISTS "LiteLLM_Table_col1_idx_123" ON "Test"("col1");'
        assert make_migration_idempotent(sql) == expected

    def test_complex_column_name(self):
        """Test column names with underscores and numbers."""
        sql = 'ALTER TABLE "Test" ADD COLUMN     "user_id_123" TEXT;'
        expected = 'ALTER TABLE "Test" ADD COLUMN IF NOT EXISTS "user_id_123" TEXT;'
        assert make_migration_idempotent(sql) == expected

    def test_empty_string(self):
        """Test empty string input."""
        assert make_migration_idempotent('') == ''
        assert make_migration_idempotent('   ') == '   '

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        sql = '\n\n  \n'
        result = make_migration_idempotent(sql)
        assert result == sql

    def test_no_matching_statements(self):
        """Test SQL with no ADD COLUMN or CREATE INDEX statements."""
        sql = '''DROP INDEX "idx1";
ALTER TABLE "Test" DROP COLUMN "col1";
SELECT * FROM "Test";'''
        result = make_migration_idempotent(sql)
        assert result == sql  # Should remain unchanged

    def test_real_world_example(self):
        """Test with a real migration file pattern."""
        sql = '''-- AlterTable
ALTER TABLE "LiteLLM_ManagedVectorStoresTable"
    ADD COLUMN     "team_id" TEXT,
    ADD COLUMN     "user_id" TEXT;

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedVectorStoresTable_team_id_idx"
    ON "LiteLLM_ManagedVectorStoresTable" ("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedVectorStoresTable_user_id_idx"
    ON "LiteLLM_ManagedVectorStoresTable" ("user_id");'''
        result = make_migration_idempotent(sql)
        assert 'ADD COLUMN IF NOT EXISTS' in result
        assert 'CREATE INDEX IF NOT EXISTS' in result
        assert result.count('IF NOT EXISTS') == 4  # 2 columns + 2 indexes

    def test_index_with_multiple_columns(self):
        """Test CREATE INDEX with multiple columns."""
        sql = 'CREATE UNIQUE INDEX "idx1" ON "Test"("col1", "col2", "col3");'
        expected = 'CREATE UNIQUE INDEX IF NOT EXISTS "idx1" ON "Test"("col1", "col2", "col3");'
        result = make_migration_idempotent(sql)
        assert 'CREATE UNIQUE INDEX IF NOT EXISTS' in result

    def test_variable_whitespace(self):
        """Test with various whitespace patterns."""
        test_cases = [
            ('ADD COLUMN "col1"', 'ADD COLUMN IF NOT EXISTS "col1"'),
            ('ADD COLUMN     "col1"', 'ADD COLUMN IF NOT EXISTS "col1"'),
            ('ADD COLUMN\t"col1"', 'ADD COLUMN IF NOT EXISTS "col1"'),
            ('ADD COLUMN\n    "col1"', 'ADD COLUMN IF NOT EXISTS "col1"'),
        ]
        for input_sql, expected_part in test_cases:
            full_sql = f'ALTER TABLE "Test" {input_sql} TEXT;'
            result = make_migration_idempotent(full_sql)
            assert expected_part in result

    def test_idempotent_function_itself(self):
        """Test that the function is idempotent (can be called multiple times)."""
        sql = 'ALTER TABLE "Test" ADD COLUMN     "col1" TEXT;'
        result1 = make_migration_idempotent(sql)
        result2 = make_migration_idempotent(result1)
        assert result1 == result2  # Should not change on second call

    def test_special_characters_in_quotes(self):
        """Test that function only matches quoted identifiers with word characters."""
        # This should NOT match (no quotes)
        sql = 'ALTER TABLE Test ADD COLUMN col1 TEXT;'
        result = make_migration_idempotent(sql)
        # Should not modify since pattern requires quotes
        assert 'IF NOT EXISTS' not in result or 'ADD COLUMN     "col1"' in sql

    def test_add_column_in_comment(self):
        """Test that ADD COLUMN in comments is not modified."""
        sql = '''-- This is a comment about ADD COLUMN
ALTER TABLE "Test" ADD COLUMN     "col1" TEXT;'''
        result = make_migration_idempotent(sql)
        # Comment should remain unchanged, but actual statement should be modified
        assert '-- This is a comment about ADD COLUMN' in result
        assert 'ADD COLUMN IF NOT EXISTS' in result

    def test_string_literal_with_add_column(self):
        """Test that ADD COLUMN in string literals is not modified."""
        # This is unlikely in migration SQL, but test for safety
        sql = 'ALTER TABLE "Test" ADD COLUMN     "col1" TEXT DEFAULT \'ADD COLUMN test\';'
        result = make_migration_idempotent(sql)
        # Should modify the ADD COLUMN statement, not the string literal
        assert 'ADD COLUMN IF NOT EXISTS' in result
        assert "'ADD COLUMN test'" in result  # String literal unchanged
