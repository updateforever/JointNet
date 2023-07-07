try:
    import wandb
except ImportError:
    raise ImportError('no wandb, run "pip install wandb" ')


class WandbWriter:
    def __init__(self, mode, name, cfg, cur_step, step_interval, id='jay'):
        self.wandb = wandb
        self.step = cur_step
        self.interval = step_interval

        wandb.init(project=mode, name=name, config=cfg, id=id)

    def write_log(self, stats):
        self.wandb.log(stats)
