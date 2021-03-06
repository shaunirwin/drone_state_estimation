# README

This is the Readme for the Python code in this project.

Install Miniconda, then:
~~~
conda env update -f python/environment.yml
~~~

Run main script:
~~~
conda activate drone_sim_env
export JAX_ENABLE_X64=True
python -m python/ekf_slam_2d.py
~~~

Run unit tests:
~~~
# ensure JAX uses 64-bit precision instead of 32-bit:
export JAX_ENABLE_X64=True
export USE_JAX=True
pytest python/unit_tests/
~~~
We need to use Jax when running unit tests, since we use automatic differentiation to check Jacobians are calculated 
correctly.