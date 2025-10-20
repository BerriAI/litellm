import sys
import os
import pytest
import ast
import ast

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router


class TestRouterIndexManagement:
    """Test cases for router index management functions"""

    @pytest.fixture
    def router(self):
        """Create a router instance for testing"""
        return Router(model_list=[])

    def test_deletion_updates_model_name_indices(self, router):
        """Test that deleting a deployment updates model_name_to_deployment_indices correctly"""
        router.model_list = [
            {"model_name": "gpt-3.5", "model_info": {"id": "model-1"}},
            {"model_name": "gpt-4", "model_info": {"id": "model-2"}},
            {"model_name": "gpt-4", "model_info": {"id": "model-3"}},
            {"model_name": "claude", "model_info": {"id": "model-4"}}
        ]
        router.model_id_to_deployment_index_map = {
            "model-1": 0, "model-2": 1, "model-3": 2, "model-4": 3
        }
        router.model_name_to_deployment_indices = {
            "gpt-3.5": [0],
            "gpt-4": [1, 2],
            "claude": [3]
        }

        # Remove one of the duplicate gpt-4 deployments
        router._update_deployment_indices_after_removal(model_id="model-2", removal_idx=1)

        # Verify indices are shifted correctly
        assert router.model_name_to_deployment_indices["gpt-3.5"] == [0]
        assert router.model_name_to_deployment_indices["gpt-4"] == [1]  # was [1,2], removed 1, shifted 2->1
        assert router.model_name_to_deployment_indices["claude"] == [2]  # was [3], shifted to [2]

        # Remove the last gpt-4 deployment
        router._update_deployment_indices_after_removal(model_id="model-3", removal_idx=1)

        # Verify gpt-4 is removed from dict when no deployments remain
        assert "gpt-4" not in router.model_name_to_deployment_indices
        assert router.model_name_to_deployment_indices["gpt-3.5"] == [0]
        assert router.model_name_to_deployment_indices["claude"] == [1]

    def test_build_model_id_to_deployment_index_map(self, router):
        """Test _build_model_id_to_deployment_index_map function"""
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "model-1"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "model-2"},
            },
        ]

        # Test: Build index from model list
        router._build_model_id_to_deployment_index_map(model_list)

        # Verify: model_list is populated
        assert len(router.model_list) == 2
        # Verify: model_id_to_deployment_index_map is correctly built
        assert router.model_id_to_deployment_index_map["model-1"] == 0
        assert router.model_id_to_deployment_index_map["model-2"] == 1

    def test_add_model_to_list_and_index_map_from_model_info(self, router):
        """Test _add_model_to_list_and_index_map extracting model_id from model_info"""
        # Setup: Empty router
        router.model_list = []
        router.model_id_to_deployment_index_map = {}
        
        # Test: Add model without explicit model_id
        model = {"model": "test-model", "model_info": {"id": "model-info-id"}}
        router._add_model_to_list_and_index_map(model=model)
        
        # Verify: Model added to list
        assert len(router.model_list) == 1
        assert router.model_list[0] == model
        
        # Verify: Index map uses model_info.id
        assert router.model_id_to_deployment_index_map["model-info-id"] == 0

    def test_add_model_to_list_and_index_map_multiple_models(self, router):
        """Test _add_model_to_list_and_index_map with multiple models to verify indexing"""
        # Setup: Empty router
        router.model_list = []
        router.model_id_to_deployment_index_map = {}
        
        # Test: Add multiple models
        model1 = {"model": "model1", "model_info": {"id": "id-1"}}
        model2 = {"model": "model2", "model_info": {"id": "id-2"}}
        model3 = {"model": "model3", "model_info": {"id": "id-3"}}
        
        router._add_model_to_list_and_index_map(model=model1, model_id="id-1")
        router._add_model_to_list_and_index_map(model=model2, model_id="id-2")
        router._add_model_to_list_and_index_map(model=model3, model_id="id-3")
        
        # Verify: All models added to list
        assert len(router.model_list) == 3
        assert router.model_list[0] == model1
        assert router.model_list[1] == model2
        assert router.model_list[2] == model3
        
        # Verify: Correct indices in map
        assert router.model_id_to_deployment_index_map["id-1"] == 0
        assert router.model_id_to_deployment_index_map["id-2"] == 1
        assert router.model_id_to_deployment_index_map["id-3"] == 2

    def test_has_model_id(self, router):
        """Test has_model_id function for O(1) membership check"""
        # Setup: Add models to router
        router.model_list = [
            {"model": "test1", "model_info": {"id": "model-1"}}, 
            {"model": "test2", "model_info": {"id": "model-2"}}, 
            {"model": "test3", "model_info": {"id": "model-3"}}
        ]
        router.model_id_to_deployment_index_map = {"model-1": 0, "model-2": 1, "model-3": 2}

        # Test: Check existing model IDs
        assert router.has_model_id("model-1") == True
        assert router.has_model_id("model-2") == True
        assert router.has_model_id("model-3") == True

        # Test: Check non-existing model IDs
        assert router.has_model_id("non-existent") == False
        assert router.has_model_id("") == False
        assert router.has_model_id("model-4") == False

        # Test: Empty router
        empty_router = Router(model_list=[])
        assert empty_router.has_model_id("any-id") == False

    def test_build_model_name_index(self, router):
        """Test _build_model_name_index function"""
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "model-1"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "model-2"},
            },
            {
                "model_name": "gpt-4",  # Duplicate model_name, different deployment
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "model-3"},
            },
        ]

        # Test: Build index from model list
        router._build_model_name_index(model_list)

        # Verify: model_name_to_deployment_indices is correctly built
        assert "gpt-3.5-turbo" in router.model_name_to_deployment_indices
        assert "gpt-4" in router.model_name_to_deployment_indices
        
        # Verify: gpt-3.5-turbo has single deployment
        assert router.model_name_to_deployment_indices["gpt-3.5-turbo"] == [0]
        
        # Verify: gpt-4 has multiple deployments
        assert router.model_name_to_deployment_indices["gpt-4"] == [1, 2]
        
        # Test: Rebuild index (should clear and rebuild)
        new_model_list = [
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "claude-3"},
                "model_info": {"id": "model-4"},
            },
        ]
        router._build_model_name_index(new_model_list)
        
        # Verify: Old entries are cleared
        assert "gpt-3.5-turbo" not in router.model_name_to_deployment_indices
        assert "gpt-4" not in router.model_name_to_deployment_indices
        
        # Verify: New entry is added
        assert "claude-3" in router.model_name_to_deployment_indices
        assert router.model_name_to_deployment_indices["claude-3"] == [0]

    def test_no_linear_scans_in_router(self):
        """
        Static analysis test to ensure Router doesn't use O(n) linear scans.
        
        Scans router.py for 'in self.model_list' pattern which indicates
        inefficient O(n) iteration instead of using index-based O(1) lookups.
        
        Methods should use:
        - model_id_to_deployment_index_map for O(1) model_id lookups
        - model_name_to_deployment_indices for O(1) + O(k) model_name lookups
        """
        # Methods that are allowed to iterate through self.model_list
        ALLOWED_METHODS = [
            "_get_deployment_by_litellm_model",  # Edge case: lookup by litellm_params.model (not indexed)
        ]
        
        # Get path to router.py
        router_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "litellm",
            "router.py"
        )
        
        # Read the file
        with open(router_file, 'r') as f:
            content = f.read()
        
        # Parse with AST
        tree = ast.parse(content)
        
        # Find violations
        violations = []
        ignore_methods = set(ALLOWED_METHODS)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_name = node.name
                
                # Skip ignored methods
                if method_name in ignore_methods:
                    continue
                
                # Get source for this method
                try:
                    method_source = ast.get_source_segment(content, node)
                    if not method_source:
                        continue
                    
                    # Check for the anti-pattern: "in self.model_list"
                    # This catches: for x in self.model_list, if x in self.model_list, etc.
                    if "in self.model_list" in method_source:
                        # Extract the specific line for better error reporting
                        lines = method_source.split('\n')
                        pattern_line = None
                        for line in lines:
                            if "in self.model_list" in line:
                                pattern_line = line.strip()
                                break
                        
                        violations.append({
                            "method": method_name,
                            "line": node.lineno,
                            "pattern": pattern_line or "in self.model_list"
                        })
                except Exception:
                    # Skip if we can't get source segment
                    pass
        
        # Assert no violations
        if violations:
            error_msg = "\n".join([
                f"  - {v['method']}() at line {v['line']}: {v['pattern']}"
                for v in violations
            ])
            
            pytest.fail(
                f"\n{'='*70}\n"
                f"Found O(n) linear scan pattern in router.py:\n\n"
                f"{error_msg}\n\n"
                f"These methods should use index maps instead:\n"
                f"  - model_id_to_deployment_index_map (for model_id lookups)\n"
                f"  - model_name_to_deployment_indices (for model_name lookups)\n\n"
                f"If a method legitimately needs O(n) iteration, add it to\n"
                f"ALLOWED_METHODS in this test method.\n"
                f"{'='*70}\n"
            )
    def test_model_names_is_set(self):
        """Verify that model_names uses a set for O(1) lookups, not a list (O(n))"""
        router = Router(model_list=[])
        
        assert isinstance(router.model_names, set), (
            f"model_names should be a set for O(1) lookups, but got {type(router.model_names)}"
        )
