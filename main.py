from aug_image.aug_image import AugImage

class AugE:
    def __init__(self, job_description: dict):
        self.path = job_description['output_path']

    def add_data(self, name: str):
        aug_img = AugImage.from_path(self.path, name)

    def finalize(self):
        pass
