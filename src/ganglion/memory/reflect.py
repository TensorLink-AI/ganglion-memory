"""Reflection utilities.

The structured experience storage that _simple_reflect used to provide
is now handled directly by agent.py's learn() method, which always fires
and stores input_text/output_text.  The counterfactual comparison in
wrap.py is the actual learning signal.

This module is kept as a namespace for future reflection helpers.
"""
