from typing import Optional
from dvclive.lightning import DVCLiveLogger as _DVCLiveLogger

class DVCLiveLogger(_DVCLiveLogger):
    """A patch to the official implementation of the PyTorch Lightning DVCLiveLogger that does not implement all the interface.
    """

    @property
    def log_dir(self) -> Optional[str]:
        return self.experiment.dir
    
    @property
    def save_dir(self) -> Optional[str]:
        return self.log_dir