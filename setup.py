from setuptools import setup, find_packages
setup(name='easydl',
      version='2.0.4',
      description='"Remove duplicate functions"',
      url='https://github.com/thuml/easydl',
      license='MIT',
      packages=['easydl','easydl.tf', 'easydl.common', 'easydl.pytorch'],
      install_requires=['torch', 'torchvision','matplotlib','pathlib2', 'six'],
      entry_points={
        'console_scripts': [
                'runTask= easydl:runTask'],
    },
      zip_safe=False)
