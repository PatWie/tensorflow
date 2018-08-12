# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Benchmarks for Cudnn RNN models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.correlation_cost.python.ops import correlation_cost_op
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class CorrelationCostOpBenchmark(test.Benchmark):
  """Benchmarks Correlation Cost layer.
  """

  def _GetTestConfig(self):
    return {
        "large": {
            "NCHW": [16, 1024, 128, 128],
        },
        "medium": {
            "NCHW": [4, 16, 16, 16],
        },
        "small": {
            "NCHW": [2, 2, 3, 4],
        },
    }

  def _GetConfigDesc(self, config):
    return "y{}".format(config["NCHW"])

  def _BenchmarkOp(self, op, desc):
    burn_in_steps = 10
    benchmark_steps = 20
    with session.Session() as sess:
      for i in xrange(burn_in_steps + benchmark_steps):
        if i == burn_in_steps:
          start_time = time.time()
        sess.run(op)
      total_time = time.time() - start_time
      step_time = total_time / benchmark_steps
      print("%s takes %.4f sec/step" % (desc, step_time))
      self.report_benchmark(
          name=desc, iters=benchmark_steps, wall_time=total_time)


  def benchmarkTfCorrelationCostForward(self):
    test_configs = self._GetTestConfig()
    for config_name, config in test_configs.items():
      NCHW = config["NCHW"]

      with ops.Graph().as_default(), ops.device("/device:GPU:0"):
        input_a = array_ops.zeros(NCHW, dtypes.float32)
        input_b = array_ops.zeros(NCHW, dtypes.float32)

        call_op = correlation_cost_op.correlation_cost
        forward_op = call_op(input_a, input_b,
                             kernel_size=1,
                             max_displacement=2,
                             stride_1=1,
                             stride_2=2,
                             pad=4,
                             data_format='NCHW')

        self._BenchmarkOp(forward_op, "tf_correlation_cost %s %s" %
                          (config_name, self._GetConfigDesc(config)))

if __name__ == "__main__":
  test.main()
