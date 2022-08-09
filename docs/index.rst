#######
Water Classification
#######

Water Classification's Module and functions
------------------------------

This package is a tool to preprocess the data, use Multiview learning/ Feature stack deep learning in order to class Water
bodies and evaluate the performance of the model.

See :`tools` folder for more information.


config.py
------------------------------
.. automodule:: tools
  :members: config

.. automodule:: tools.config
  :members: configuration

metrics\_.py
------------------------------
.. automodule:: tools
  :members: metrics\_

.. automodule:: tools.metrics_
  :members: f1, custom_accuracy, MetricCalculator, MetricCalculator_multiview_2, MetricCalculator_multiview_3, MetricCalculator_NDWI, ndwi_threashold
  :noindex: metrics\_

model.py
------------------------------
.. automodule:: tools
  :members: model

.. automodule:: tools.model
  :members: conv_block, EncoderMiniBlock, DecoderMiniBlock, CustomModel, get_model, CustomModel_multiview_2, get_model_multiview_2, CustomModel_multiview_3, get_model_multiview_3, get_model_multiview_2_HT
  :noindex: model

preprocessing.py
------------------------------
.. automodule:: tools
  :members: preprocessing

.. automodule:: tools.preprocessing
  :members: Preprocessor, maskL8sr, EnsureTwodigit, GenSeasonalDatesMonthly
  :noindex: preprocessing

sampling.py
------------------------------
.. automodule:: tools
  :members: sampling

.. automodule:: tools.sampling
  :members: Training_task, Eval_task, Testing_task
  :noindex: sampling

images.py
------------------------------
.. automodule:: tools
  :members: images

.. automodule:: tools.images
  :members: doExport, predictionSingleinput, predictionMultipleinput, predictionMultipleinput_3, uploadToGEEAsset, doPrediction_featurestack, doPrediction_multiview_2, doPrediction_multiview_3, LoadImage
  :noindex: images

losses\_.py
------------------------------
.. automodule:: tools
  :members: losses\_

.. automodule:: tools.losses_
  :members: dice_coef, dice_p_cc
  :noindex: losses\_

.. rubric:: References

