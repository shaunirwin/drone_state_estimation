def test_jax_using_64_bit_precision():
    from jax.config import config

    # check that JAX is set to 64-bit precision
    assert config.values["jax_enable_x64"]
