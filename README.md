![Screenshot](logo.PNG)

> Geographic Information Systems for Electrification

# About

The GIS for electrification (Gisele) tool was developed as an effort to improve the planning of rural electrification in developing countries. It is an open source Python-based tool that uses GIS and terrain analysis to model the area under study, groups loads using a density-based clustering algorithm called DBSCAN and then it uses graph theory to find the least-costly electric network topology that can connect all the people in the area. 

The methodology of Gisele consists in three main steps: data analysis, clustering and grid routing. During the initial phase of data gathering and analysis, GIS data sets are created in order to properly map several information about the area to be electrified. Some of these information are: population density, elevation, slope and roads. They are all processed using a weighting strategy that translates the topological aspect of the terrain to the difficulty of line deployment. Then, DBSCAN is used to strategically aggregates groups of people in small areas called clusters. The output is a set number of clusters, partially covering the initial area considered, in which the grid routing algorithm is performed. Finally, Gisele uses the concept of Steiner tree to create a network topology connecting all the aggregated people in each cluster, and then, if necessary, it makes use of Dijkstra’s algorithm to connect each cluster grid into an existing distribution network.

# Requirements
* Python 3.7
* Solver for MILP optimization: the default is 'gurobi'

# Getting started
Once having downloaded Python and cloned/download the project, it is possible to automatically create the environment with the useful packages by running in the command prompt:

```
conda env create -f environment.yml
```
Run 
```
Gisele.py
```
Gisele is provided by a user interface which can be accessed by clicking on the link that appears on the console, or directly opening the page http://127.0.0.1:8050/ in a web browser.
For more information see the documentation in Gisele/docs

# Documentation

# Contributing

# Citing 

# Licencing

# GISEle code structure rules

## General structure:

* Maximum Line Length: 79 characters for coding lines and 72 characters to docstrings and comments.

* Tab indentation (4 spaces).

* Break lines before binary operators (+, - ).

* Imports on the top of the code following the hierarchy: standard library, third-party, local.

* Single quote for code strings and double quote for docstrings.

* Don't compare boolean values to True or False using ==:

* For sequences, (strings, lists, tuples), use the fact that empty sequences are false:

## Blank lines:

* Surround top-level function and class definitions with two blank lines.

* Method definitions inside a class are surrounded by a single blank line.

* Extra blank lines may be used (sparingly) to separate groups of related functions. Blank lines may be omitted between a bunch of related one-liners (e.g. a set of dummy implementations).

* Use blank lines in functions, sparingly, to indicate logical sections.

## Comments:

* Comments that contradict the code are worse than no comments. Always make a priority of keeping the comments up-to-date when the code changes!

* Comments should be complete sentences. The first word should be capitalized, unless it is an identifier that begins with a lower case letter (never alter the case of identifiers!).

* You should use two spaces after a sentence-ending period in multi- sentence comments, except after the final sentence.

* Block comments generally apply to some (or all) code that follows them, and are indented to the same level as that code. Each line of a block comment starts with a # and a single space (unless it is indented text inside the comment).

* Paragraphs inside a block comment are separated by a line containing a single #.

* Use inline comments sparingly. An inline comment is a comment on the same line as a statement. Inline comments should be separated by at least two spaces from the statement. They should start with a # and a single space.

* Inline comments are unnecessary and in fact distracting if they state the obvious.


## Docstrings:

* Write docstrings for all public modules, functions, classes, and methods. Docstrings are not necessary for non-public methods, but you should have a comment that describes what the method does. This comment should appear after the def line.

* Note that most importantly, the """ that ends a multiline docstring should be on a line by itself

# Consistency is key!



<img src="./images/logo.sample.png" alt="Logo of the project" align="right">

# Name of the project &middot; [![Build Status](https://img.shields.io/travis/npm/npm/latest.svg?style=flat-square)](https://travis-ci.org/npm/npm) [![npm](https://img.shields.io/npm/v/npm.svg?style=flat-square)](https://www.npmjs.com/package/npm) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/your/your-project/blob/master/LICENSE)
> Additional information or tag line

A brief description of your project, what it is used for.

## Installing / Getting started

A quick introduction of the minimal setup you need to get a hello world up &
running.

```shell
commands here
```

Here you should say what actually happens when you execute the code above.

## Developing

### Built With
List main libraries, frameworks used including versions (React, Angular etc...)

### Prerequisites
What is needed to set up the dev environment. For instance, global dependencies or any other tools. include download links.


### Setting up Dev

Here's a brief intro about what a developer must do in order to start developing
the project further:

```shell
git clone https://github.com/your/your-project.git
cd your-project/
packagemanager install
```

And state what happens step-by-step. If there is any virtual environment, local server or database feeder needed, explain here.

### Building

If your project needs some additional steps for the developer to build the
project after some code changes, state them here. for example:

```shell
./configure
make
make install
```

Here again you should state what actually happens when the code above gets
executed.

### Deploying / Publishing
give instructions on how to build and release a new version
In case there's some step you have to take that publishes this project to a
server, this is the right time to state it.

```shell
packagemanager deploy your-project -s server.com -u username -p password
```

And again you'd need to tell what the previous code actually does.

## Versioning

We can maybe use [SemVer](http://semver.org/) for versioning. For the versions available, see the [link to tags on this repository](/tags).


## Configuration

Here you should write what are all of the configurations a user can enter when using the project.

## Tests

Describe and show how to run the tests with code examples.
Explain what these tests test and why.

```shell
Give an example
```

## Style guide

Explain your code style and show how to check it.

## Api Reference

If the api is external, link to api documentation. If not describe your api including authentication methods as well as explaining all the endpoints with their required parameters.


## Database

Explaining what database (and version) has been used. Provide download links.
Documents your database design and schemas, relations etc... 

## Licensing

State what the license is and how to find the text version of the license.


