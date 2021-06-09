# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from abc import ABC
import io
import base64
import torch
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from io import BytesIO
from .base_handler import BaseHandler


class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """
    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True

    def _get_img(self, row):
        """Compat layer: normally the envelope should just return the data
        directly, but older version of KFServing envelope and Torchserve in general
        didn't have things set up right
        """

        if isinstance(row, dict):
            image = row.get("data") or row.get("body")
        else:
            image = row
            
        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)

        return image

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            image = self._get_img(row)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)


    def get_insights(self, tensor_data, _, target=0):
        # TODO: this will work with Image Classification, but needs work for segmentation
        all_attr = self.ig.attribute(tensor_data, target=target, n_steps=15)
        n, c, h, w = all_attr.size()
        reshape_attr = all_attr.view(n, h, w, c).detach().cpu().numpy()    
        return_bytes = []
        for attr in reshape_attr:
            matplot_viz, _ = viz.visualize_image_attr(attr, use_pyplot=False, sign='all')
            
            fout = BytesIO()
            matplot_viz.savefig(fout)
            return_bytes.append(fout.getvalue())
            
        return return_bytes