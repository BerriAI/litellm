import pytest
import os
import tempfile
import importlib.util
from unittest.mock import patch, MagicMock
from litellm.proxy.types_utils.utils import get_instance_fn, _load_instance_from_remote_storage


class TestCustomLoggerS3GCS:
    """Test custom logger loading from S3/GCS using URL prefixes"""

    @pytest.fixture
    def sample_custom_logger_content(self):
        """Sample custom logger file content"""
        return '''
from litellm.integrations.custom_logger import CustomLogger

class TestCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.initialized = True
    
    def log_pre_api_call(self, model, messages, kwargs):
        print(f"Pre-API call to {model}")

# Instance to be imported
test_logger_instance = TestCustomLogger()
'''

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_local_file_loading_still_works(self, temp_config_dir, sample_custom_logger_content):
        """Test that local file loading continues to work (no URL prefix)"""
        # Create a local custom logger file
        custom_logger_path = os.path.join(temp_config_dir, "test_custom_logger.py")
        with open(custom_logger_path, 'w') as f:
            f.write(sample_custom_logger_content)
        
        # Create a dummy config file
        config_path = os.path.join(temp_config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("model_list: []")
        
        # Test loading the custom logger (traditional way)
        instance = get_instance_fn("test_custom_logger.test_logger_instance", config_path)
        
        assert instance is not None
        assert hasattr(instance, 'initialized')
        assert instance.initialized is True

    def test_s3_url_parsing(self):
        """Test S3 URL parsing"""
        test_url = "s3://my-bucket/loggers/custom_callbacks.proxy_handler_instance"
        
        # Mock the download function to avoid actual S3 calls
        with patch('litellm.proxy.common_utils.load_config_utils.download_python_file_from_s3') as mock_download:
            mock_download.return_value = False  # Will cause failure, but we just want to test parsing
            
            with pytest.raises(ImportError, match="Failed to download"):
                _load_instance_from_remote_storage(test_url)
            
            # Verify the download was called with correct parameters
            mock_download.assert_called_once()
            call_args = mock_download.call_args
            assert call_args.kwargs['bucket_name'] == "my-bucket"
            assert call_args.kwargs['object_key'] == "loggers/custom_callbacks.py"

    def test_gcs_url_parsing(self):
        """Test GCS URL parsing"""
        test_url = "gcs://my-bucket/custom_logger.my_instance"
        
        # Mock the download function
        with patch('litellm.proxy.types_utils.utils._download_gcs_file_wrapper') as mock_download:
            mock_download.return_value = False  # Will cause failure
            
            with pytest.raises(ImportError, match="Failed to download"):
                _load_instance_from_remote_storage(test_url)
            
            # Verify the download was called with correct parameters
            mock_download.assert_called_once()
            call_args = mock_download.call_args
            assert call_args[0][0] == "my-bucket"  # bucket_name (positional for _download_gcs_file_wrapper)
            assert call_args[0][1] == "custom_logger.py"  # object_key

    @patch('litellm.proxy.common_utils.load_config_utils.download_python_file_from_s3')
    def test_s3_download_success(self, mock_s3_download, sample_custom_logger_content):
        """Test successful S3 download and loading"""
        # Configure S3 download to succeed and create the file
        def mock_download(bucket_name, object_key, local_file_path):
            with open(local_file_path, 'w') as f:
                f.write(sample_custom_logger_content)
            return True
        
        mock_s3_download.side_effect = mock_download
        
        # Test loading with S3 URL
        test_url = "s3://test-bucket/test_custom_logger.test_logger_instance"
        instance = get_instance_fn(test_url)
        
        assert instance is not None
        assert hasattr(instance, 'initialized')
        assert instance.initialized is True
        
        # Verify S3 download was called with correct parameters
        mock_s3_download.assert_called_once()
        call_args = mock_s3_download.call_args
        assert call_args.kwargs['bucket_name'] == 'test-bucket'
        assert call_args.kwargs['object_key'] == 'test_custom_logger.py'

    @patch('litellm.proxy.types_utils.utils._download_gcs_file_wrapper')
    def test_gcs_download_success(self, mock_gcs_download, sample_custom_logger_content):
        """Test successful GCS download and loading"""
        # Configure GCS download to succeed and create the file
        def mock_download(bucket_name, object_key, local_file_path):
            with open(local_file_path, 'w') as f:
                f.write(sample_custom_logger_content)
            return True
        
        mock_gcs_download.side_effect = mock_download
        
        # Test loading with GCS URL
        test_url = "gcs://test-bucket/test_custom_logger.test_logger_instance"
        instance = get_instance_fn(test_url)
        
        assert instance is not None
        assert hasattr(instance, 'initialized')
        assert instance.initialized is True

    def test_nested_path_parsing(self):
        """Test parsing of nested paths in URLs"""
        test_url = "s3://my-bucket/loggers/production/advanced_logger.handler_instance"
        
        with patch('litellm.proxy.common_utils.load_config_utils.download_python_file_from_s3') as mock_download:
            mock_download.return_value = False
            
            with pytest.raises(ImportError):
                _load_instance_from_remote_storage(test_url)
            
            # Verify correct object key was generated
            call_args = mock_download.call_args
            assert call_args.kwargs['object_key'] == "loggers/production/advanced_logger.py"

    def test_invalid_url_schemes(self):
        """Test error handling for invalid URL schemes"""
        # URLs that look like URLs but aren't s3:// or gcs:// will be treated as module names
        # and fail with regular ImportError
        with pytest.raises(ImportError):
            get_instance_fn("http://bucket/module.instance")
        
        with pytest.raises(ImportError):
            get_instance_fn("ftp://bucket/module.instance")

    def test_invalid_url_format(self):
        """Test error handling for invalid URL formats"""
        # Missing bucket
        with pytest.raises(ImportError, match="Invalid URL format"):
            get_instance_fn("s3://")
        
        # Missing path
        with pytest.raises(ImportError, match="Invalid URL format"):
            get_instance_fn("s3://bucket-only")
        
        # Missing instance name
        with pytest.raises(ImportError, match="Invalid module specification"):
            get_instance_fn("s3://bucket/module-only")
            
        # Including .py extension (common mistake)
        with pytest.raises(ImportError, match="Don't include '\\.py' extension and you must specify the instance name"):
            get_instance_fn("s3://bucket/custom_guardrail.py")

    @patch('litellm.proxy.common_utils.load_config_utils.download_python_file_from_s3')
    def test_download_failure_handling(self, mock_s3_download):
        """Test handling of download failures"""
        mock_s3_download.return_value = False
        
        test_url = "s3://test-bucket/failing_logger.instance"
        
        with pytest.raises(ImportError, match="Failed to download"):
            get_instance_fn(test_url)

    @patch('litellm.proxy.common_utils.load_config_utils.download_python_file_from_s3')
    def test_file_cleanup(self, mock_s3_download, sample_custom_logger_content):
        """Test that temporary files are cleaned up"""
        created_files = []
        
        def mock_download(bucket_name, object_key, local_file_path):
            created_files.append(local_file_path)
            with open(local_file_path, 'w') as f:
                f.write(sample_custom_logger_content)
            return True
        
        mock_s3_download.side_effect = mock_download
        
        test_url = "s3://test-bucket/test_custom_logger.test_logger_instance"
        instance = get_instance_fn(test_url)
        
        assert instance is not None
        
        # Verify file was created and then cleaned up
        assert len(created_files) == 1
        temp_file = created_files[0]
        assert not os.path.exists(temp_file), f"Temporary file {temp_file} was not cleaned up"

    def test_no_url_prefix_fallback(self, temp_config_dir):
        """Test fallback when no URL prefix is used and local file doesn't exist"""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("model_list: []")
        
        # Test that it tries local loading when no URL prefix is used
        with pytest.raises(ImportError, match="Could not import instance from nonexistent_logger"):
            get_instance_fn("nonexistent_logger.instance", config_path) 