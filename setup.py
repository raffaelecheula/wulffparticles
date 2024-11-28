from setuptools import setup, find_packages

setup(
    name="wulffparticles",
    version="0.0.1",
    url="https://github.com/raffaelecheula/wulffparticles.git",
    author="Raffaele Cheula",
    author_email="cheula.raffaele@gmail.com",
    description="Tools to create atomistic models of nanoparticles.",
    license="GPL-3.0",
    install_requires=find_packages(),
    python_requires=">=3.5, <4",
)