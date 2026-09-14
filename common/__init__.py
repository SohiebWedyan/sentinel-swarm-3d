"""Shared, dependency-light utilities used across every layer.

Nothing in this package should import from perception/, navigation/, swarm/,
etc. — it sits below all of them so those layers can freely import from here
without circular imports.
"""
