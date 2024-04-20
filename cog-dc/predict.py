from cog import BasePredictor, Input, Path
import tempfile

from oot_diffusion import OOTDiffusionModel

# full body


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = OOTDiffusionModel(None, None, "dc")
        self.model.load_pipe()

        return self.model

    # The arguments and types the model takes as input
    def predict(
        self,
        model_image: Path = Input(
            description="Clear picture of the model",
            default="https://raw.githubusercontent.com/qiweiii/oot_diffusion/main/oot_diffusion/assets/model_8.png",
        ),
        garment_image: Path = Input(
            description="Clear picture of upper body garment",
            default="https://raw.githubusercontent.com/qiweiii/oot_diffusion/main/oot_diffusion/assets/dress_1.jpg",
        ),
        garment_category: str = Input(
            description="Garment category: upperbody, lowerbody, dress",
            default="upperbody",
            choices=["upperbody", "lowerbody", "dress"],
        ),
        steps: int = Input(
            default=20, description="Inference steps", ge=1, le=40),
        guidance_scale: float = Input(
            default=2.0, description="Guidance scale", ge=1.0, le=5.0
        ),
        seed: int = Input(default=0, description="Seed",
                          ge=0, le=0xFFFFFFFFFFFFFFFF),
    ) -> list[Path]:
        """Run a single prediction on the model"""

        generated_images, mask = self.model.generate(
            model_path=model_image,
            cloth_path=garment_image,
            steps=steps,
            cfg=guidance_scale,
            seed=seed,
            category=garment_category,
            num_samples=4,
        )

        result_paths: list[Path] = []

        mask_path = Path(tempfile.mktemp(suffix=".png"))
        mask.save(mask_path, "PNG")
        result_paths.append(mask_path)

        for i, img in enumerate(generated_images):
            result_path = Path(tempfile.mktemp(suffix=".png"))
            img.save(result_path, "PNG")
            result_paths.append(result_path)

        return result_paths
