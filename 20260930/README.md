# 20260930

Stepping through a minimal Sphinx project to get a detailed understanding of
how the HTML build works.

## Quickstart

1. `cd` into the repo containing this README.

2. `../bzl run :sphinx.run`

   You should see output like this:

   ```
   INFO: Analyzed target //:sphinx.run (1 packages loaded, 1919 targets configured).
   INFO: Found 1 target...
   Target //:sphinx.run up-to-date:
     bazel-bin/sphinx.run
   INFO: Elapsed time: 3.768s, Critical Path: 0.03s
   INFO: 2 processes: 8 action cache hit, 2 internal.
   INFO: Build completed successfully, 2 total actions
   INFO: Running command line: bazel-bin/sphinx.run
   + exec env -- bin/build --show-traceback --builder=html --fail-on-warning _sphinx/_sources /tmp/sphinx-out
   > …/.cache/bazel/_bazel_kayce/27c9ba2ca7e59497d50aa80e8c762914/execroot/_main/bazel-out/k8-fastbuild/bin/sphinx.run.runfiles/_main/build.py(17)<module>()
   -> main()
   (Pdb)
   ```

## Development

### Update Python dependencies

1. Edit `pypi.txt`.
2. `../bzl run //:pypi.update`
