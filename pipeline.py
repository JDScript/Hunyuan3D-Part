from pathlib import Path

from partgen.partformer_pipeline import PartFormerPipeline
from partgen.utils.misc import get_config_from_file
from partgen.utils.misc import instantiate_from_config
from safetensors import torch as safetensors
import torch


class XPartPipeline(PartFormerPipeline):
    @classmethod
    def from_pretrained(
        cls,
        config: dict,
        dtype=torch.float32,
        ignore_keys=(),
        device="cuda",
        **kwargs,
    ):
        # Initialize model
        model = instantiate_from_config(config["model"])
        vae = instantiate_from_config(config["shapevae"])
        conditioner = vae
        if config.get("conditioner") is not None:
            conditioner = instantiate_from_config(config["conditioner"])
        scheduler = instantiate_from_config(config["scheduler"])
        bbox_predictor = instantiate_from_config(config["bbox_predictor"])

        # Load weights
        model.load_state_dict(
            safetensors.load_file(Path(config["ckpt_path"]) / "model" / "model.safetensors"),
        )
        vae.load_state_dict(
            safetensors.load_file(Path(config["ckpt_path"]) / "shapevae" / "shapevae.safetensors"),
        )
        if config.get("conditioner") is not None:
            conditioner.load_state_dict(
                safetensors.load_file(
                    Path(config["ckpt_path"]) / "conditioner" / "conditioner.safetensors",
                ),
            )
        bbox_predictor.model.load_state_dict(
            state_dict=safetensors.load_file(
                Path(config["ckpt_path"]) / "p3sam" / "p3sam.safetensors",
            ),
        )

        model_kwargs = {
            "vae": vae,
            "model": model,
            "scheduler": scheduler,
            "conditioner": conditioner,
            "bbox_predictor": bbox_predictor,
            "device": device,
            "dtype": dtype,
        }
        model_kwargs.update(kwargs)

        return cls(**model_kwargs)


if __name__ == "__main__":
    pipeline = XPartPipeline.from_pretrained(
        config=get_config_from_file(
            Path(__file__).parent / "XPart" / "partgen" / "config" / "infer.yaml",
        )
    )
