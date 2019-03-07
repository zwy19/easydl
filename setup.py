from setuptools import setup, find_packages
setup(name='easydl',
      version='2.0.0',
      description='"Only support PyTorch"',
      url='https://github.com/thuml/easydl',
      license='MIT',
      packages=['easydl','easydl.tf', 'easydl.common', 'easydl.pytorch'],
      install_requires=['torch', 'torchvision','tensorpack','matplotlib','pathlib2', 'docopt', 'six'],
      entry_points={
        'console_scripts': [
                'runTask= easydl:runTask'],
    },
      zip_safe=False)
