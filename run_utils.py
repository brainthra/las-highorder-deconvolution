class Resolution:
    def __init__(self, Bmin, Bmax, Bstep, Btype):
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.Bstep = Bstep
        self.Btype = Btype


class EnergyLevel:
    def __init__(self, Emin, Emax, target_bins, ext_factor):
        self.Emin = Emin
        self.Emax = Emax
        self.target_bins = target_bins
        self.ext_factor = ext_factor

class ModelConfig:
    def __init__(self, model_type, pred_bins_target, pred_tar_scale, pred_bins_ext, pred_ext_scale):
        self.model_type = model_type
        self.pred_bins_target = pred_bins_target
        self.pred_tar_scale = pred_tar_scale
        self.pred_bins_ext = pred_bins_ext
        self.pred_ext_scale = pred_ext_scale

class NoiseLevel:
    def __init__(self, noise_min, noise_max):
        self.noise_min = noise_min
        self.noise_max = noise_max

class FluxLevel:
    def __init__(self, flux_min, flux_max):
        self.flux_min = flux_min
        self.flux_max = flux_max


class Condition:
    def __init__(self,resolution, noise_level, flux_level):
        self.resolution = resolution
        self.noise_level = noise_level
        self.flux_level = flux_level