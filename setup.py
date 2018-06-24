from setuptools import setup, find_packages
setup(name='easydl',
      version='1.2.0',
      description='"adjust Dataset hierarchy, change default epsilon, add Accumulator"',
      url='https://github.com/thuml/easydl',
      license='MIT',
      packages=['easydl','easydl.tf', 'easydl.common', 'easydl.pytorch'],
      install_requires=['tensorflow','tensorlayer','tensorpack','matplotlib','pathlib2'],
      entry_points={
        'console_scripts': [
                'runTask= easydl:runTask'],
    },
      zip_safe=False)
