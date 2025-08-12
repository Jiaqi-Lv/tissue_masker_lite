from setuptools import find_packages, setup

setup(
    name="tissue_masker_lite",
    version="0.1",
    description="A Lightweight, quality-control model for generating tissue masks",
    url="https://github.com/Jiaqi-Lv/tissue_masker_lite",
    author="Jiaqi Lv",
    author_email="jiaqi.lv@warwick.ac.uk",
    license="BSD 3-Clause License",
    packages=find_packages(),
    # install_requires=["tiatoolbox", "segmentation-models-pytorch"],
    zip_safe=False,
    include_package_data=True,          
    package_data={"tissue_masker_lite": ["model_weights/*.pth"]},
)
