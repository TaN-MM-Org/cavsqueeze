# Contributing to cavsqueeze

Thank you for your interest in improving `cavsqueeze`. Contributions of every size are welcome: bug reports, questions about the physics conventions, documentation fixes, new test cases, and new features.

## Reporting problems and asking for help

Open an issue at https://github.com/TaN-MM-Org/cavsqueeze/issues. For bug reports, please include the package version (`python -c "import cavsqueeze; print(cavsqueeze.__version__)"`), a minimal script that reproduces the problem, and the output you expected. Questions about usage and about the model conventions are welcome in issues as well; there is no separate forum.

## Development setup

```
git clone https://github.com/TaN-MM-Org/cavsqueeze
cd cavsqueeze
pip install -e .[test]
pytest tests -q
```

The test suite validates the cumulant solver against exact QuTiP and PIQS references and takes about two minutes on a laptop. All tests must pass before a pull request is merged, and the same suite runs in CI on every push and pull request.

## Pull requests

Fork the repository, create a branch, and open a pull request against `main`. Please keep the following in mind:

* Every change to the physics must come with a test against an exact reference, a closed-form limit, or an independent solver (the repository already contains QuTiP, PIQS and discrete truncated Wigner references to test against).
* Pure NumPy/SciPy is preferred in the core package; QuTiP is an optional dependency used only in `cavsqueeze.exact` and in the tests.
* Docstrings state the conventions (operator ordering, frame, units); changes that touch conventions must update the docstrings in the same commit.
* American spelling in prose.

## Governance and support

The package is maintained by Tanvir Mahmud Mahim (BRAC University), who reviews issues and pull requests. The scientific content is developed with the co-authors of the associated paper. Releases are tagged on GitHub and published to PyPI by CI.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0 that covers the project.
