#from dipy.direction.probabilistic_direction_getter import ProbabilisticDirectionGetter
from dipy.tracking.direction_getter import DirectionGetter
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

from dipy.reconst import shm

from dipy.direction.pmf import PmfGen
from dipy.direction.peaks import peak_directions, default_sphere

from dipy.tracking.stopping_criterion import (
    TRACKPOINT, OUTSIDEIMAGE, ENDPOINT)

from random import random

import torch

import numpy as np

def to_torch(arr_like, dev):
    return torch.from_numpy(np.asarray(arr_like)).to(
            device=dev, dtype=torch.float32)


def trilinear_interpolate4d_torch(
        data,
        point,
        result,
        index,
        weight,
        d_shape):
    """
    Tri-linear interpolation along the last dimension of a 4d array

    Parameters
    ----------
    point : 1d array (3,)
        3 doubles representing a 3d point in space. If point has integer values
        ``[i, j, k]``, the result will be the same as ``data[i, j, k]``.
    data : 4d array
        Data to be interpolated.
    result : 1d array
        The result of interpolation. Should have length equal to the
        ``data.shape[3]``.
    Returns
    -------
    point or None on failure

    """
    if data.shape[3] != result.shape[0]:
        return False
    if torch.any(point < -.5):
        return False
    if torch.any(point >= d_shape - .5):
        return False

    flr = torch.floor(point)
    rem = point - flr

    index[:, 0] = flr + (flr == -1)
    index[:, 1] = flr + (flr != (d_shape - 1))
    weight[:, 0] = 1 - rem
    weight[:, 1] = rem

    N = data.shape[3]
    # Generate the weights w using broadcasting
    w = weight[0][:, None, None, None] * weight[1][None, :, None, None] * weight[2][None, None, :, None]

    # Expand index to match the broadcasting
    expanded_index = index[:, :, None].expand(-1, -1, N)

    # Gather data based on expanded_index
    gathered_data = data[expanded_index[0], expanded_index[1], expanded_index[2], torch.arange(N)]

    # Compute the weighted sum
    result[:] = torch.sum(w * gathered_data, dim=(0, 1, 2))

    return True

class TorchSHCoeffPmfGen(PmfGen):
    def __init__(self,
                 shcoeff_array,
                 sphere,
                 basis_type,
                 legacy=True,
                 dev="cpu"):
        self.data = np.asarray(shcoeff_array, dtype=float)
        self.sphere = sphere
        self.dev = dev

        sh_order = shm.order_from_ncoef(self.data.shape[3])
        try:
            basis = shm.sph_harm_lookup[basis_type]
        except KeyError:
            raise ValueError("%s is not a known basis type." % basis_type)
        B, _, _ = basis(sh_order, sphere.theta, sphere.phi, legacy=legacy)

        self.data_on_dev = to_torch(self.data, dev)
        self.B_on_dev = to_torch(B, dev)

        self.coeff_on_dev = torch.empty(
            self.data.shape[3], device=dev, dtype=torch.float32)
        self.pmf_on_dev = torch.empty(
            B.shape[0], device=dev, dtype=torch.float32)
        self.index = torch.zeros((3, 2), dtype=torch.int, device=dev)
        self.weight = torch.zeros((3, 2), dtype=torch.float32, device=dev)
        self.d_shape = to_torch(self.data.shape[:3], dev)

    def get_pmf(self, point):
        if not trilinear_interpolate4d_torch(
                self.data_on_dev, point, self.coeff_on_dev,
                self.index, self.weight, self.d_shape):
            self.pmf_on_dev = 0.0
        else:
            self.pmf_on_dev = torch.matmul(
                self.B_on_dev, self.coeff_on_dev)
        return self.pmf_on_dev

class TorchProbabilisticDirectionGetter(DirectionGetter):
    def __init__(self, pmf_gen, pmf_threshold, sphere, max_angle):
        self.pmf_threshold = pmf_threshold
        self.pmf_gen = pmf_gen
        self.sphere = sphere

        self.cos_similarity = to_torch(
            np.cos(np.deg2rad(max_angle)), pmf_gen.dev)
        self.vertices = to_torch(
            self.sphere.vertices.copy(), pmf_gen.dev)

    @classmethod
    def from_pmf(*args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_shcoeff(cls, shcoeff, max_angle, sphere=default_sphere,
                     pmf_threshold=0.1, basis_type=None, legacy=True, **kwargs):
        """Probabilistic direction getter from a distribution of directions
        on the sphere

        Parameters
        ----------
        shcoeff : array
            The distribution of tracking directions at each voxel represented
            as a function on the sphere using the real spherical harmonic
            basis. For example the FOD of the Constrained Spherical
            Deconvolution model can be used this way. This distribution will
            be discretized using ``sphere`` and tracking directions will be
            chosen from the vertices of ``sphere`` based on the distribution.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        basis_type : name of basis
            The basis that ``shcoeff`` are associated with.
            ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        legacy: bool, optional
            True to use a legacy basis definition for backward compatibility
            with previous ``tournier07`` and ``descoteaux07`` implementations.

        See Also
        --------
        dipy.direction.peaks.peak_directions

        """
        pmf_gen = TorchSHCoeffPmfGen(
            shcoeff, sphere,
            basis_type, legacy=legacy, dev=kwargs.get("dev", "cpu"))
        return cls(pmf_gen, pmf_threshold, sphere, max_angle)


    def _get_pmf(self, point):
        pmf = self.pmf_gen.get_pmf(point[:3])
        absolute_pmf_threshold = self.pmf_threshold*torch.max(pmf)
        pmf[pmf<absolute_pmf_threshold] = 0.0
        return pmf

    def initial_direction(self, point):
        point = to_torch(point, self.pmf_gen.dev)
        pmf = self._get_pmf(point).numpy(force=True).astype(np.double)
        return peak_directions(pmf, self.sphere)[0]

    def get_direction_torch(self, point, direction):
        """Samples a pmf to updates ``direction`` array with a new direction.

        Parameters
        ----------
        point : memory-view (or ndarray), shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : memory-view (or ndarray), shape (3,)
            Previous tracking direction.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.

        """
        pmf = self._get_pmf(point)

        # print(self._adj_matrix.keys())
        # bool_array = self._adj_matrix[
        #     (direction[0], direction[1], direction[2])]
        bool_array = torch.matmul(direction, self.vertices.T)

        pmf[torch.abs(bool_array) >= self.cos_similarity] = 0.0
        pmf = torch.cumsum(pmf, 0)

        last_cdf = pmf[-1]
        if last_cdf == 0:
            return False

        random_sample = random() * last_cdf
        idx = (random_sample >= pmf).nonzero(as_tuple=False)
        if idx.numel() > 0:
            idx = idx[-1, 0].item() + 1
        else:
            idx = 0

        newdir = self.vertices[idx, :]
        # Update direction and return 0 for error
        if direction[0] * newdir[0] \
         + direction[1] * newdir[1] \
         + direction[2] * newdir[2] > 0:

            direction[0] = newdir[0]
            direction[1] = newdir[1]
            direction[2] = newdir[2]
        else:
            direction[0] = -newdir[0]
            direction[1] = -newdir[1]
            direction[2] = -newdir[2]
        return True

    def fixed_step_torch(self, point, direction, step_size):
        point[:] += direction * step_size

    def step_to_boundary_torch(self, point, direction, overstep):
        raise NotImplementedError()

    def generate_streamline(self, seed, direction, voxel_size,
                            step_size, stopping_criterion,
                            streamline, stream_status,
                            fixedstep):
        if not isinstance(
                stopping_criterion,
                ThresholdStoppingCriterion):
            raise NotImplementedError(
                "TorchProbabilisticDirectionGetter can only currently be used"
                " with ThresholdStoppingCriterion")

        if fixedstep > 0:
            step = self.fixed_step_torch
        else:
            step = self.step_to_boundary_torch
        dev = self.pmf_gen.dev

        direction = to_torch(direction, dev)
        voxel_size = to_torch(voxel_size, dev)

        threshold = to_torch(stopping_criterion.threshold, dev)
        metric_map = np.asarray(stopping_criterion.metric_map)
        sc = to_torch(metric_map[..., np.newaxis], dev)
        sc_res = torch.empty((sc.shape[3]), device=dev, dtype=torch.float32)
        d_shape = to_torch(metric_map.shape[:3], dev)

        point = to_torch(seed, dev)
        streamline_on_gpu = torch.empty(
            streamline.shape, device=dev, dtype=torch.float32)
        streamline_on_gpu[0, :] = point

        stream_status = TRACKPOINT
        for i in range(1, streamline.shape[0]):
            if not self.get_direction_torch(point, direction):
                break

            voxdir = direction / voxel_size

            step(point, voxdir, step_size)
            streamline_on_gpu[i, :] = point

            if not trilinear_interpolate4d_torch(
                    sc,
                    point,
                    sc_res,
                    self.pmf_gen.index,
                    self.pmf_gen.weight,
                    d_shape):
                stream_status = OUTSIDEIMAGE
                break

            if sc_res[0] > threshold:
                continue
            else:
                stream_status = ENDPOINT
                break
        else:
            # maximum length of streamline has been reached, return everything
            i = streamline.shape[0]
        streamline[:i] = streamline_on_gpu[:i].numpy(force=True)
        return i, stream_status