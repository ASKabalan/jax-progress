import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_cpu_devices():
    """Configure JAX to use 8 CPU devices for shard_map testing."""
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_num_cpu_devices", 8)
    yield


@pytest.fixture
def skip_if_insufficient_devices():
    """Skip tests if insufficient devices available (less than 8)."""
    if len(jax.devices()) < 8:
        pytest.skip("Requires 8 devices for shard_map tests")
