import autolens as al


class main:

    def __init__(self, masked_dataset, transformer_class=al.TransformerFINUFFT):

        self.masked_dataset = masked_dataset

        self.transformer = transformer_class(
            uv_wavelengths=self.masked_dataset.uv_wavelengths,
            grid=self.masked_dataset.grid.in_radians
        )
