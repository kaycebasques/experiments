import importlib.metadata
import os
import pdb
import sys
import sysconfig

from sphinx.cmd.build import main


def _load_repos() -> dict[str, str]:
    repos = {}
    stdlib = os.path.abspath(sysconfig.get_path("stdlib")) + os.sep
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    repos[stdlib] = f"https://github.com/python/cpython/blob/v{py_ver}/Lib"

    for dist in importlib.metadata.distributions():
        version = dist.metadata["Version"]
        repo = None
        for entry in dist.metadata.get_all("Project-URL", []):
            label, url = [part.strip() for part in entry.split(",", 1)]
            if label.lower() in ("code", "source", "repository"):
                repo = url.rstrip("/")
                break
        if not repo:
            homepage = dist.metadata.get("Home-page", "")
            if "github.com" in homepage:
                repo = homepage.rstrip("/")
        if repo:
            site_pkgs = os.path.abspath(dist.locate_file("")) + os.sep
            repos[site_pkgs] = f"{repo}/blob/v{version}"
    return repos


class GithubPdb(pdb.Pdb):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._repos = _load_repos()

    def format_stack_entry(self, frame_lineno, lprefix=": "):
        entry = super().format_stack_entry(frame_lineno, lprefix)
        frame, lineno = frame_lineno
        filename = self.canonic(frame.f_code.co_filename)
        for prefix, base_url in self._repos.items():
            if filename.startswith(prefix):
                rel_path = filename.removeprefix(prefix)
                url = f"{base_url}/{rel_path}#L{lineno}"
                if lprefix in entry:
                    head, tail = entry.split(lprefix, 1)
                    return f"{head}\n> {url}{lprefix}{tail}"
                return f"{entry}\n> {url}"
        return entry


sys.breakpointhook = lambda *args, **kws: GithubPdb().set_trace(
    sys._getframe().f_back
)


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
