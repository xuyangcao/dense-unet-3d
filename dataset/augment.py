import numpy as np 
import random
import torch


class ElasticTransform(object):
    r"""
    Elastic transform using b-spline interplation
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if round(np.random.uniform(0, 1), 1) <= self.probability:
            num_control_points = np.random.randint(20, 32)
            image, label = self._produceRandomlyDeformedImage(image, label, num_control_points)

        sample['image'], sample['label'] = image, label
        return sample


    def _produceRandomlyDeformedImage(self, image, label, num_control_points=15, std=2, seed=1):
        r""" This function comes from V-net，deform a image by B-spine interpolation

        Arguments:
        ---
        image: numpy array
            input image 

        label: numpy array
            binary mask of the input image

        num_control_points: int. Defalut: 15 
            control point，B-spine interpolation parameters

        std: int. Dafault: 2
            Deviation，B-spine interpolation parameters

        Returns:
        ---
        out_image: ndarray
            deformed image

        out_label: ndarray
            deformed label
        """

        sitkImage = sitk.GetImageFromArray(image, isVector=False)
        sitklabel = sitk.GetImageFromArray(label, isVector=False)

        transform_domain_mesh_size = [num_control_points] * sitkImage.GetDimension()

        tx = sitk.BSplineTransformInitializer(
            sitkImage, transform_domain_mesh_size)

        params = tx.GetParameters()

        params_np = np.asarray(params, dtype=float)
        np.random.seed(seed)
        params_np = params_np + np.random.randn(params_np.shape[0]) * std

        # remove z deformations! The resolution in z is too bad
        params_np[0:int(len(params) / 3)] = 0
        params = tuple(params_np)
        tx.SetParameters(params)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitkImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(tx)

        resampler.SetDefaultPixelValue(0)
        out_ing_sitk = resampler.Execute(sitkImage)
        out_label_sitk = resampler.Execute(sitklabel)

        out_img = sitk.GetArrayFromImage(out_ing_sitk)
        out_img = out_img.astype(dtype=np.float32)
        out_label = sitk.GetArrayFromImage(out_label_sitk)

        return out_img, out_label



class RandomFlip(object):
    r""" random flip image along single and multi-axis.

    """
    def __init__(self, probability=0.5):
        self.probability = probability
        self.flip_dict = [
                [0],
                [1],
                [2],
                [0, 1],
                [0, 2],
                [1, 2],
                [0, 1, 2]
                ] 

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if round(np.random.uniform(0, 1), 1) <= self.probability:
            idx = random.randint(0, len(self.flip_dict) - 1)
            image = np.flip(image, axis=self.flip_dict[idx]).copy()
            label = np.flip(label, axis=self.flip_dict[idx]).copy()

        sample['image'], sample['label'] = image, label
        return sample


class RandomNoise(object):
    r""" add noise to input image

    """
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise

        return {'image': image, 'label': label}


class ToTensor(object):
    r"""Convert ndarrays in sample to Tensors.

    """
    def __init__(self, use_dismap=False):
        self.use_dismap = use_dismap

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample['image'] = torch.from_numpy(image)
        #print('label.shape: ', label.shape)
        sample['label'] = torch.from_numpy(label).long()

        if self.use_dismap:
            dis_map = sample['dis_map']
            dis_map = np.expand_dims(dis_map, 0)
            #print('dis_map.shape: ', dis_map.shape)
            sample['dis_map'] = torch.from_numpy(dis_map.astype(np.float32))

        return sample
