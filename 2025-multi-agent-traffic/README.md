## Makefile
Check the Makefile. It will help you. There are a lot commands.

## Setup project
This command will install you pyenv on yout system. You have to source the .bashrc file after running that command:
```bash
make install-uv
```

To setup python (install python version and all deps + init pre-commit):
```bash
make setup
```

If you want to clean up python in the project:
```bash
make clean-python
```

Linting the repo with:
```bash
uvx ruff check --fix
```

You can fomat the repo with:
```bash
uvx ruff format
```

You can also run for linting and formatting:
```bash
make format
```

To run tox and pytest:
```bash
uvx tox
```
