import queue
import numpy as np
from ..utils import assert_import, record_import_error
from ._masker import Masker
from .._serializable import Serializer, Deserializer

try:
    import cv2
except ImportError as e:
    record_import_error("cv2", "cv2 could not be imported!", e)


class Image(Masker):
    """ This masks out image regions with blurring or inpainting.
    """

    def __init__(self, mask_value, shape=None):
        """ Build a new Image masker with the given masking value.

        Parameters
        ----------
        mask_value : np.array, "blur(kernel_xsize, kernel_xsize)", "inpaint_telea", or "inpaint_ns"
            The value used to mask hidden regions of the image.

        shape : None or tuple
            If the mask_value is an auto-generated masker instead of a dataset then the input
            image shape needs to be provided.
        """

        # [KRELL]
        print(">> Enter masker._image.py: __init__")
        print("Input mask_value =", mask_value)
        print("Input shape = ", shape)

        if shape is None:
            if isinstance(mask_value, str):
                raise TypeError("When the mask_value is a string the shape parameter must be given!")
            self.input_shape = mask_value.shape # the (1,) is because we only return a single masked sample to average over
        else:
            self.input_shape = shape

        self.input_mask_value = mask_value

        # This is the shape of the masks we expect
        self.shape = (1, np.prod(self.input_shape)) # the (1, ...) is because we only return a single masked sample to average over

        # [KRELL]
        print("The masker will accept masks of shape = ", self.shape)
        print("     Why shape[0] = 1? --> 'we only return a single masked sample to average over'")
        print("     Shape[1] is just flattend input: {}x{}x{} = {}".format(shape[0], shape[1], shape[2], self.shape[1]))

        self.blur_kernel = None
        self._blur_value_cache = None
        if issubclass(type(mask_value), np.ndarray):
            self.mask_value = mask_value.flatten()
        elif isinstance(mask_value, str):
            assert_import("cv2")

            self.mask_value = mask_value
            if mask_value.startswith("blur("):
                self.blur_kernel = tuple(map(int, mask_value[5:-1].split(",")))

             # [KRELL]
            print("Using OpenCV for masking")
            print("Created blue kernel from input: ", self.blur_kernel)

        else:
            self.mask_value = np.ones(self.input_shape).flatten() * mask_value

        self.build_partition_tree(mode=2)

        # note if this masker can use different background for different samples
        self.fixed_background = not isinstance(self.mask_value, str)

        # [KRELL]
        print("Fixed background?", self.fixed_background)
        print("    (Not really sure what fixed background means..)")

        #self.scratch_mask = np.zeros(self.input_shape[:-1], dtype=np.bool)
        self.last_xid = None

        # [KRELL]
        print("<< Exit masker._image.py: __init__")

    def __call__(self, mask, x, verbose=False):

        # [KRELL]
        if verbose:
            print("\n        >> Enter masker/_image.py: __call__")

        if np.prod(x.shape) != np.prod(self.input_shape):
            raise Exception("The length of the image to be masked must match the shape given in the " + \
                            "ImageMasker contructor: "+" * ".join([str(i) for i in x.shape])+ \
                            " != "+" * ".join([str(i) for i in self.input_shape]))

        # unwrap single element lists (which are how single input models look in multi-input format)
        if isinstance(x, list) and len(x) == 1:
            x = x[0]

        # we preserve flattend inputs as flattened and full-shaped inputs as their original shape
        in_shape = x.shape
        if len(x.shape) > 1:
            x = x.flatten()

        # if mask is not given then we mask the whole image
        if mask is None:
            mask = np.zeros(np.prod(x.shape), dtype=np.bool)

        if isinstance(self.mask_value, str):
            if self.blur_kernel is not None:
                if self.last_xid != id(x):
                    self._blur_value_cache = cv2.blur(x.reshape(self.input_shape), self.blur_kernel).flatten()
                    self.last_xid = id(x)

                    # [KRELL]
                    if verbose:
                        print("        Before  blur: {}".format(x))
                        print("        Applied blur  {}  on image of shape {}.".format(self.blur_kernel, x.reshape(self.input_shape).shape))
                        print("        After   blur: {}".format(self._blur_value_cache))

                out = x.copy()
                out[~mask] = self._blur_value_cache[~mask]

                # [KRELL]
                if verbose:
                    print("        Replace masked values with blurred values")
                    print("        Mask: {}".format(mask))
                    print("        Output image: {}".format(out))

            elif self.mask_value == "inpaint_telea":
                out = self.inpaint(x, ~mask, "INPAINT_TELEA")
            elif self.mask_value == "inpaint_ns":
                out = self.inpaint(x, ~mask, "INPAINT_NS")
        else:
            out = x.copy()
            out[~mask] = self.mask_value[~mask]

        # [KRELL]
        if verbose:
            print("        << Exit masker/_image.py: __call__\n")

        return (out.reshape(1, *in_shape),)

    def inpaint(self, x, mask, method):
        """ Fill in the masked parts of the image through inpainting.
        """
        reshaped_mask = mask.reshape(self.input_shape).astype(np.uint8).max(2)
        if reshaped_mask.sum() == np.prod(self.input_shape[:-1]):
            out = x.reshape(self.input_shape).copy()
            out[:] = out.mean((0, 1))
            return out.flatten()

        return cv2.inpaint(
            x.reshape(self.input_shape).astype(np.uint8),
            reshaped_mask,
            inpaintRadius=3,
            flags=getattr(cv2, method)
        ).astype(x.dtype).flatten()

    def build_partition_tree(self, mode=1):
        """ This partitions an image into a herarchical clustering based on axis-aligned splits.
        """

        # [KRELL]
        print("\n  >> Enter masker._image.py: build_partition_tree")

        xmin = 0
        xmax = self.input_shape[0]
        ymin = 0
        ymax = self.input_shape[1]
        zmin = 0
        zmax = self.input_shape[2]

        # [KRELL]
        print("    x_range = [{}, {}],   y_range = [{}, {}],   z_range = [{}, {}]".format(
            xmin, xmax, ymin, ymax, zmin, zmax))
        print("      (Here x:height,  y:width,  z:channels)")

        #total_xwidth = xmax - xmin
        total_ywidth = ymax - ymin
        total_zwidth = zmax - zmin
        q = queue.PriorityQueue()

        M = int((xmax - xmin) * (ymax - ymin) * (zmax - zmin))
        self.clustering = np.zeros((M - 1, 4))

        # [KRELL]
        print("    Create hierarchical clustering of size {},".format(self.clustering.shape))
        print("      where dim 0 is {}x{}x{}={}".format(xmax, zmax, ymax, self.clustering.shape))
        print("      and dim 1 is 4 because each entry has:  [left child, right child, cost, subtree size]")

        q.put((0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False))

        # [KRELL]
        print("    Init priority queue q")
        print("      Each entry:  [priority, xmin, xmax, ymin, ymax, zmin, zmax, parent idx, is left?]")

        ind = len(self.clustering) - 1
        while not q.empty():
            _, xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left = q.get()

            # [KRELL]
            if ind == len(self.clustering) - 1 or ind == 0:
                print("    Current, ", ("", xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left))

            if parent_ind >= 0:
                self.clustering[parent_ind, 0 if is_left else 1] = ind + M

                # [KRELL]
                # print("Place the current parition index into parent's child slot")
                # print("    If a left child -> in slot 0,  right child -> in slot 1")

            # make sure we line up with a flattened indexing scheme
            if ind < 0:
                assert -ind - 1 == xmin * total_ywidth * total_zwidth + ymin * total_zwidth + zmin

            xwidth = xmax - xmin
            ywidth = ymax - ymin
            zwidth = zmax - zmin
            if xwidth == 1 and ywidth == 1 and zwidth == 1:
                pass
            else:

                # by default our ranges remain unchanged
                lxmin = rxmin = xmin
                lxmax = rxmax = xmax
                lymin = rymin = ymin
                lymax = rymax = ymax
                lzmin = rzmin = zmin
                lzmax = rzmax = zmax

                # Mode 0: default image partition scheme where channel-wise is last
                if mode == 0:

                    # split the xaxis if it is the largest dimension
                    if xwidth >= ywidth and xwidth > 1:
                        xmid = xmin + xwidth // 2
                        lxmax = xmid
                        rxmin = xmid

                    # split the yaxis
                    elif ywidth > 1:
                        ymid = ymin + ywidth // 2
                        lymax = ymid
                        rymin = ymid

                    # split the zaxis only when the other ranges are already width 1
                    else:
                        zmid = zmin + zwidth // 2
                        lzmax = zmid
                        rzmin = zmid


                # Mode 1: cubes (split x, y, z axes in order)
                if mode == 1:

                    # split the xaxis if it is the largest dimension
                    if xwidth >= ywidth and xwidth > 1:
                        xmid = xmin + xwidth // 2
                        lxmax = xmid
                        rxmin = xmid

                    # split the yaxis
                    elif ywidth >= zwidth and ywidth > 1:
                        ymid = ymin + ywidth // 2
                        lymax = ymid
                        rymin = ymid

                    else:
                        zmid = zmin + zwidth // 2
                        lzmax = zmid
                        rzmin = zmid

                # Mode 3: split bands first
                if mode == 2:

                    # split the zaxis if it is larger than 1
                    if zwidth > 1:
                        zmid = zmin + zwidth // 2
                        lzmax = zmid
                        rzmin = zmid

                    # split the xaxis if it is the largest dimension
                    elif xwidth >= ywidth and xwidth > 1:
                        xmid = xmin + xwidth // 2
                        lxmax = xmid
                        rxmin = xmid

                    # split the yaxis
                    elif ywidth > 1:
                        ymid = ymin + ywidth // 2
                        lymax = ymid
                        rymin = ymid


                lsize = (lxmax - lxmin) * (lymax - lymin) * (lzmax - lzmin)
                rsize = (rxmax - rxmin) * (rymax - rymin) * (rzmax - rzmin)

                q.put((-lsize, lxmin, lxmax, lymin, lymax, lzmin, lzmax, ind, True))
                q.put((-rsize, rxmin, rxmax, rymin, rymax, rzmin, rzmax, ind, False))

                # [KRELL]
                if ind == len(self.clustering) - 1 or ind == 0:
                    print("    Left child", (-lsize, lxmin, lxmax, lymin, lymax, lzmin, lzmax, ind, True))
                    print("    Right child:", (-rsize, rxmin, rxmax, rymin, rymax, rzmin, rzmax, ind, False))
                    print("    -----------")

            ind -= 1

        # fill in the group sizes
        for i in range(len(self.clustering)):
            li = int(self.clustering[i, 0])
            ri = int(self.clustering[i, 1])
            lsize = 1 if li < M else self.clustering[li-M, 3]
            rsize = 1 if ri < M else self.clustering[ri-M, 3]
            self.clustering[i, 3] = lsize + rsize

        # [KRELL]
        # In the above, loop through cluster entries and set the size of each subtree
        # based on summing each subtree's left and right tree sizes (accumulate up to the top)

        # [KRELL]
        print("  Generated partition hierarchy:")
        print(self.clustering)
        print("  shape = {}".format(self.clustering.shape))
        print("  << Exit masker._image.py: build_partition_tree \n")

    def save(self, out_file):
        """ Write a Image masker to a file stream.
        """
        super().save(out_file)

        # Increment the verison number when the encoding changes!
        with Serializer(out_file, "shap.maskers.Image", version=0) as s:
            s.save("mask_value", self.input_mask_value)
            s.save("shape", self.input_shape)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """ Load a Image masker from a file stream.
        """
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.Image", min_version=0, max_version=0) as s:
            kwargs["mask_value"] = s.load("mask_value")
            kwargs["shape"] = s.load("shape")
        return kwargs
