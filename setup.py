from setuptools import find_packages, setup

reqs = [
    'numpy<=1.24',
    'numba',
    'scipy',
    'pandas',
    'scikit-learn',
    'scikit-image',
    'matplotlib',
    'pyOpenSSL',
    'datashader',
    'pip',
    'pillow',
    'opencv-python',
    'imageio'
]

docs_extras = [
    'Sphinx >= 6.2.1',
    'sphinx-design',
    'pydata-sphinx-theme',
    'numpydoc'
]

setup(
    name='clustersort',
    version='0.0.1',
    description='Spike sorting and processing utilities.',
    author='Flynn OConnell',
    author_email='Flynnoconnell@gmail.com',
    url='https://www.github.com/Flynnoconnell/clustersort',
    packages=find_packages(where='clustersort'),
    package_dir={'': 'clustersort'},
    install_requires=reqs,
    extras_require={'docs': docs_extras},
    include_package_data=True,
    zip_safe=False,
    classifiers = [
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    python_requires='>=3.9,<3.10',
)
