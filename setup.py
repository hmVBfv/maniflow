from distutils.core import setup

setup(name='ManiFlow',
      version='1.0',
      description='Python library that implements some algorithms to interact with manifolds',
      author='Yangshan Xiang, Mark Robin Riegraf, Minming Zhao, Felix Widmaier',
      url='https://gitlab.gwdg.de/yangshan.xiang/scientific-computing/',
      packages=["maniflow"],
      package_dir={'maniflow': "maniflow/"},
      options={'bdist_wheel': {'universal': '1'}},
      )
