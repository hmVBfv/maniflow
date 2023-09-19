from setuptools import setup, find_packages

setup(name='maniflow',
      version='1.0',
      description='Python library that implements some algorithms to interact with manifolds',
      author='Yangshan Xiang, Mark Robin Riegraf, Minming Zhao, Felix Widmaier',
      url='https://gitlab.gwdg.de/yangshan.xiang/scientific-computing/',
      packages=find_packages(include=['maniflow', 'maniflow.*', 'maniflow.mesh.*', 'maniflow.globals']),
      package_data={'maniflow': ['maniflow/globals/data.json']},
      include_package_data=True,
      install_requires=['numpy'],
      options={'bdist_wheel': {'universal': '1'}},
      )
