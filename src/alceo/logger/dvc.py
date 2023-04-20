from typing import Optional
from dvclive.lightning import DVCLiveLogger as _DVCLiveLogger

class DVCLiveLogger(_DVCLiveLogger):
    @property
    def log_dir(self) -> Optional[str]:
        return self.experiment.dir
    
    @property
    def save_dir(self) -> Optional[str]:
        return self.log_dir