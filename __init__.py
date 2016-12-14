# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# #############################################################################
# This is an implementaion of Convolutional Neural Netwroks for 3D volumes.
# The code is a modification of the code provided in TensorFlow tutorials for
# CIFAR-10 dataset.
# For help, please refer to the 'readme' file
# The code is modifed by: Mohamed ElMikaty
# Last update: 14 Dec 16
################################################################################

"""Makes helper libraries available"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import net_model
import in_data
import configure
