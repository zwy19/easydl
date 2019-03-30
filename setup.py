from setuptools import setup, find_packages
setup(name='easydl',
      version='2.0.6',
      description='"bugfix in one_hot"',
      url='https://github.com/thuml/easydl',
      license='MIT',
      packages=['easydl','easydl.tf', 'easydl.common', 'easydl.pytorch'],
      install_requires=['torch', 'torchvision','matplotlib','pathlib2', 'six'],
      entry_points={
        'console_scripts': [
                'runTask= easydl:runTask'],
    },
      zip_safe=False)
