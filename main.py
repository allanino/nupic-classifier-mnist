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
import os

from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager

import model_params
import pickle

_LOGGER = logging.getLogger(__name__)

_NUM_RECORDS = 42000
_TRAIN_SIZE = int(_NUM_RECORDS * 0.8)

def createModel():
  return ModelFactory.create(model_params.MODEL_PARAMS)

def runHotgym():
  count = 0 # Count test set
  correct = 0 # Count correct predictions in test set

  model = createModel()
  model.enableInference({'predictedField': 'label'})
  with open('train.csv') as fin:
    reader = csv.reader(fin)
    headers = reader.next()

    for i, record in enumerate(reader, start=1):
      modelInput = dict(zip(headers, record))
      modelInput["label"] = str(modelInput["label"])
      for j in range(0,784):
        modelInput["pixel%d" % j] = int(modelInput["pixel%d" % j])

      # Disable learning to calculate accuracy in test set
      if i == _TRAIN_SIZE:
        model.disableLearning()

      result = model.run(modelInput)

      # Calculate accuracy of test set
      if i >= _TRAIN_SIZE:
        print "Label:",modelInput["label"]
        print "Predicted:", result.inferences['multiStepBestPredictions'][0]
        print result.inferences['multiStepPredictions'][0]
        if modelInput["label"] == result.inferences['multiStepBestPredictions'][0]:
          correct = correct + 1
          print 'Status: correct'
        else:
          print 'Status: wrong'
        print
        count = count + 1

        _LOGGER.info("Predicting: %d",count)
      else:
        _LOGGER.info("Training: %d",i)

      isLast = i == _NUM_RECORDS
      if isLast:
        break

  # Return the calculated accuracy
  return float(correct)/count

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  ac = runHotgym()
  print "\nAccuracy = %.3f" % ac