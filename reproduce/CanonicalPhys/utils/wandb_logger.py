"""Optional Weights & Biases experiment tracking.

Usage in trainers:
    from utils.wandb_logger import WandbLogger
    wb = WandbLogger(config, project="emergentphys")
    wb.log({"train/loss": loss, "epoch": epoch})
    wb.log_config(config)
    wb.finish()

If wandb is not installed, all methods are no-ops.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class WandbLogger:
    """Lightweight wrapper for W&B logging. No-ops if wandb is unavailable."""

    def __init__(self, config=None, project="emergentphys", name=None,
                 enabled=True):
        """Initialize W&B run.

        Args:
            config: Config object or dict to log as run config.
            project: W&B project name.
            name: W&B run name (auto-generated if None).
            enabled: Set False to disable even if wandb is installed.
        """
        self.enabled = enabled and _WANDB_AVAILABLE

        if not _WANDB_AVAILABLE and enabled:
            logger.info("wandb not installed. Install with: pip install wandb")

        if self.enabled:
            try:
                config_dict = self._to_dict(config) if config else None
                wandb.init(project=project, name=name, config=config_dict)
                logger.info(f"W&B run initialized: {wandb.run.name}")
            except Exception as e:
                logger.warning(f"W&B init failed: {e}. Disabling W&B logging.")
                self.enabled = False

    def log(self, metrics, step=None):
        """Log metrics dict to W&B."""
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_config(self, config):
        """Update run config."""
        if self.enabled:
            wandb.config.update(self._to_dict(config), allow_val_change=True)

    def log_artifact(self, path, name, artifact_type="model"):
        """Log a file as a W&B artifact."""
        if self.enabled:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def watch(self, model, log="gradients", log_freq=100):
        """Watch model gradients/parameters."""
        if self.enabled:
            wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()

    @staticmethod
    def _to_dict(config):
        """Convert config object to dict."""
        if isinstance(config, dict):
            return config
        if hasattr(config, '__dict__'):
            result = {}
            for k, v in config.__dict__.items():
                if hasattr(v, '__dict__'):
                    result[k] = WandbLogger._to_dict(v)
                else:
                    result[k] = v
            return result
        return {}
