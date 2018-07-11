from setuptools import setup, find_packages
setup(name='easydl',
      version='1.3.0',
      description='"add gan evaluation(fid and inception score)"',
      url='https://github.com/thuml/easydl',
      license='MIT',
      packages=['easydl','easydl.tf', 'easydl.common', 'easydl.pytorch'],
      install_requires=['tensorflow','tensorlayer','tensorpack','matplotlib','pathlib2', 'docopt'],
      entry_points={
        'console_scripts': [
                'runTask= easydl:runTask'],
    },
      zip_safe=False)
