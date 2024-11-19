from setuptools import setup, find_packages

setup(
    name="phase-sc",  
    version="2.0.2",  
    author="Qinhua Wu", 
    author_email="wuqinhua21@mails.ucas.ac.cn",  
    description="PHASE:PHenotype prediction with Attention mechanisms for Single-cell Exploring", 
    long_description=open("README.md").read(), 
    long_description_content_type="text/markdown",  
    url="https://github.com/wuqinhua/PHASE.git",
    packages=find_packages(), 
    classifiers=[
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent", 
    ],
    python_requires=">=3.10", 
    entry_points={
        "console_scripts": [
            "PHASEtrain=PHASE.main:main",  
        ],
    },
)
