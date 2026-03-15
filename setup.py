from setuptools import setup, find_packages

setup(
    name="ppocr-vectron",
    version="0.1.0",
    description="A lightweight OCR vectorization engine by VectorElectron",
    author="vectorelectron",
    packages=find_packages(),
    package_data={
        "ppocr_vectron": ["model/*"],
    },
    include_package_data=True,
    install_requires=[
        "onnxruntime",
        "numpy",
    ],
    python_requires='>=3.8',
)
