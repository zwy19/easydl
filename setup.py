from setuptools import setup, find_packages
setup(name='easydl',
      version='0.1.0',
      description='"document ready for use"',
      url='https://github.com/thuml/easydl',
      license='MIT',
      packages=['easydl','easydl.tf', 'easydl.common', 'easydl.pytorch'],
      install_requires=['tensorflow','tensorlayer','tensorpack','matplotlib','pathlib2','pytreebank'],
      entry_points={
        'console_scripts': [
                'runTask= easydl:runTask'],
    },
      zip_safe=False)
