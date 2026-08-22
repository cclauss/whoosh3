# Security Policy

## Supported versions

Security fixes are applied to the latest released version of `whoosh3` on
PyPI. Older `3.x` releases are not backported; please upgrade to the current
release before reporting an issue.

| Version        | Supported          |
| -------------- | ------------------ |
| Latest `3.x`   | :white_check_mark: |
| Older releases | :x:                |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Use GitHub's [private vulnerability reporting][pvr] for this repository
(the **"Report a vulnerability"** button under the *Security* tab). This keeps
the report confidential while it is being triaged.

[pvr]: https://github.com/priya-sundaram-dev/whoosh/security/advisories/new

When reporting, please include:

- the `whoosh3` version and Python version,
- a minimal snippet or description that reproduces the issue, and
- the impact you have observed or expect.

## What to expect

- **Acknowledgement:** I aim to confirm receipt within 7 days.
- **Assessment:** I will investigate and let you know whether the report is
  accepted, along with a rough timeline for a fix.
- **Disclosure:** once a fix is released, the advisory is published and
  reporters are credited (unless you prefer to remain anonymous).

## Scope

Whoosh is a pure-Python library that indexes and searches data you provide.
Note that, like most search and serialization libraries, **opening an index
built by an untrusted third party is not a supported trust boundary** — index
files are a data format, not a sandbox. Reports about processing deliberately
malformed indexes are welcome as robustness bugs, but treat index files from
untrusted sources with the same caution you would any untrusted input.
