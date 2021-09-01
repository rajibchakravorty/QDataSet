"""Set up for qmldataset package
"""

import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="qmldataset",
    version="0.1.0",
    author="Rajib Chakravorty",
    author_email="4748396+rajibchakravorty@users.noreply.github.com",
    description="Synthetic data generator for ML applications in Quantum Hardware system",
    long_description="",
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    #project_urls={
    #    "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    #},
    #classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    #],
    #package_dir={"": "src"},
    packages=setuptools.find_packages(include=['qmldataset', 'qmldataset.*']),
    python_requires=">=3.8, <4.0"
)