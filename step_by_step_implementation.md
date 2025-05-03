# step 1: project configuration

## install pipx and Install Poetry with pipx

pipx allows for isolated installation of Python applications.
Poetry is our dependency management tool

```
pip install pipx
pipx ensurepath

pipx install poetry
poetry --version

poetry init
// Add Project Dependencies
# Core dependencies
poetry add tensorflow scikit-learn PyYAML matplotlib Pillow

# Development dependencies
poetry add pytest ipython --group dev

```

## Create Virtual Environment

Create virtual environment and install dependencies

```
poetry install
poetry lock
poetry env activate


# Check installed packages
poetry show

# Add new packages
poetry add package-name

# Update dependencies
poetry update

# Export requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt
``` 