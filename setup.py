import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="potto",
    version="0.0.1",
    author="Jesse Michel",
    author_email="jmmichel@mit.edu",
    description="just a humble potto",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={'': ['']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.14.5',
        'typed-argument-parser',
        'dearpygui>=1.3.1',
    ]
)
