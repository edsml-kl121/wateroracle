try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='tools',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Environments for water classification.",
      long_description="Environment for Kl-121.",
      packages=['tools'])
