#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""A simple client to create a classifier for MNIST dataset"""

import csv
import datetime
import logging

from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager

import model_params
import pickle

_LOGGER = logging.getLogger(__name__)

_NUM_RECORDS = 40000

def createModel():
  return ModelFactory.create(model_params.MODEL_PARAMS)

def runHotgym():
  acc = [0] # list of accuracy
  count = 1
  model = createModel()
  model.enableInference({'predictedField': 'label'})

  with open('train.csv') as fin:
    reader = csv.reader(fin)
    headers = reader.next()

    for i, record in enumerate(reader, start=1):
      modelInput = dict(zip(headers, record))
      modelInput["label"] = int(modelInput["label"])
      for j in range(0,784):
        modelInput["pixel%d" % j] = int(modelInput["pixel%d" % j])

      if i == 32000:
        model.disableLearning()

      result = model.run(modelInput)

      if i > 32000:
        if modelInput["label"] == int(result.inferences['multiStepBestPredictions'][0] + 0.5):
            ac = (acc[count-1]*(count-1.) + 1.)/count
        else:
            ac = (acc[count-1]*(count-1.))/count

        acc.append(ac)

        _LOGGER.info("%d: %.4f", count, ac)

        count = count + 1

      isLast = i == _NUM_RECORDS
      if isLast:
        break
  return acc

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  acc = runHotgym()
  pickle.dump(acc, open('acc.p', 'wb'))