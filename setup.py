from setuptools import setup, find_packages

setup(
    name='ocr-faster-rcnn',
    version='0.0.1',
    url='https://github.com/ChenXiaoTemp/ocr-faster-rcnn',
    packages=find_packages(exclude=['examples', 'tests']),
    install_requires=[
        'pandas==0.22.0;python_version=="3.5"',
        'pandas>=0.22.0;python_version>"3.5"',
        'numpy>=1.14.2',
        'websocket_client==0.32.0',
        'numba==0.28.1;python_version=="3.5"',
        'numba>=0.28.1;python_version>"3.5"',
        'fastparquet',
        'ballast==0.4.0',
        "opencv-python",
        'torch',
        'torchvision',
        'pyyaml',
        'jsonpath-ng<=1.4.3',
        'coloredlogs',
        'IPython',
        'ipdb',
        'sh', 'flask'
    ],
    setup_requires=['wheel'],
    zip_safe=False,
    python_requires='>=3.5.2',
)
