import setuptools

setuptools.setup(
    name='theseusqfls',
    version='1.0',
    license='MIT',
    description='GPU-accelerated curve fitting using nonlinear optimization from Theseus',
    url='https://github.com/cbchosy/theseusqfls',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'theseus-ai',
        'torch',
        'h5py',
        'numpy',
        'pandas',
        'tqdm',
        'scipy'
    ]
)