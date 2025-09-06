# Agent Instructions for `vitra`

This document provides instructions for AI agents working on the `vitra` repository.

## Project Overview

`vitra` is a molecular energy and minimizer, capable of reading from and writing to `.pdb`, files.

## Development Setup

This project uses Poetry for dependency management. To set up the development environment, follow these steps:

1.  **Install dependencies:**
    ```bash
    poetry install
    ```
2.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

## Running Tests

The tests are written using `pytest`. To run the test suite, use the following command:

```bash
poetry run pytest
```

## Linting

This project uses `ruff` for linting. To check the code for style issues, run:

```bash
poetry run ruff check .
```

To automatically fix linting issues, run:

```bash
poetry run ruff check . --fix
```
