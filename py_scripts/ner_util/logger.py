import wandb

class Logger:
    """Wandb logger for tracking model training and evaluation."""
    
    def __init__(self, project, config, name):
        # start a new wandb run to track this script

        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config=config,
            # give this run a name
            name=name,
        )
    
    def __call__(self, values):
        # log the values to wandb (e.g. {"acc": acc, "loss": loss})
        wandb.log(values)