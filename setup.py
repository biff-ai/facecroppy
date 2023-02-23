from setuptools import setup

setup(
    name='facecroppy',
    version='0.1.0',
    author='Christophe Carvenius',
    author_email='christophe@biff.ai',
    description='A library for smart face cropping from images',
    packages=['facecroppy'],
    install_requires=[
        'mtcnn>=0.1.1',
        'opencv-python>=4.1.0',
        'numpy>=1.14.5',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
