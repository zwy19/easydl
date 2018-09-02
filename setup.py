from setuptools import setup, find_packages
setup(name='easydl',
      version='1.8.0',
      description='"multiple data / label support for dataset"',
      url='https://github.com/thuml/easydl',
      license='MIT',
      packages=['easydl','easydl.tf', 'easydl.common', 'easydl.pytorch'],
      install_requires=['tensorflow','tensorlayer','tensorpack','matplotlib','pathlib2', 'docopt', 'six'],
      entry_points={
        'console_scripts': [
                'runTask= easydl:runTask'],
    },
      zip_safe=False)
