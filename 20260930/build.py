import sys

from sphinx.cmd.build import main


if __name__ == "__main__":
    args = []
    for arg in sys.argv:
        if arg.startswith("@"):
            with open(arg.removeprefix("@")) as fp:
                lines = [line.strip() for line in fp if line.strip()]
            args.extend(lines)
        else:
            args.append(arg)
    sys.argv[:] = args
    breakpoint()
    main()
