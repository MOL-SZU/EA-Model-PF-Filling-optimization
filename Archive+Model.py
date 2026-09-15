"""Legacy filename retained as a launcher for the package implementation.

The former monolithic experiment, clustering, and multi-head model have
been removed. Use ``python "Archive+Model.py" run --config ...`` or the
installed ``ea-model run`` command.
"""

from ea_model.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

