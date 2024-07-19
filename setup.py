from setuptools import setup

setup(
    name='healjax',
    version='0.2.2',
    url='https://github.com/ghcollin/healjax',
    description='Healpix routines for JAX.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords='jax healpix healpy',
    license='MIT',
    author='ghcollin',
    author_email='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=['healjax'],
    install_requires=['numpy']
)