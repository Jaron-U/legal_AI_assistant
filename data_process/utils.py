from typing import Any
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# load the data file from the directory
class LawLoader(DirectoryLoader):
    """Load law books."""
    def __init__(self, path: str, **kwargs: Any) -> None:
        loader_cls = TextLoader
        glob = "**/*.md"
        super().__init__(path, loader_cls=loader_cls, glob=glob, **kwargs)

class LawLoaderTXT(DirectoryLoader):
    """Load law books."""
    def __init__(self, path: str, **kwargs: Any) -> None:
        loader_cls = TextLoader
        glob = "*.txt"
        super().__init__(path, loader_cls=loader_cls, glob=glob, **kwargs)