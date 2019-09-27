import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='dokr',  
     version='0.1',
     scripts=['dokr'] ,
     author="Jesse Bartola",
     author_email="jrbartola@gmail.com",
     description="A machine learning fairness toolkit",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/brittjay0104/fairkit-learn",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
