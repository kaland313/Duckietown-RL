"""
Utilities for displaying salient objects maps. The method used in these functions is explained in:
Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car
https://arxiv.org/abs/1704.07911
"""
__author__ = "András Kalapos"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import numpy as np
import cv2
import tensorflow as tf


def find_layer_by_name(model, name):
    layer_idx = -1
    for i, layer in enumerate(model.layers):
        if layer.name == name:
            layer_idx = i
    return layer_idx


def nvidia_salient_map(model: tf.keras.Model, obs, output_vector_idx=None):
    """
    Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car
    https://arxiv.org/abs/1704.07911
    """
    layer_outputs = [model.layers[find_layer_by_name(model, 'conv_out')].output,
                     model.layers[find_layer_by_name(model, 'conv3')].output,
                     model.layers[find_layer_by_name(model, 'conv2')].output,
                     model.layers[find_layer_by_name(model, 'conv1')].output]
    model_partial = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)
    activations = model_partial.predict(obs[None, ...])

    if output_vector_idx is None:
        salient_map = np.average(activations[0][0, :, :, :], axis=2)
    else:
        salient_map = activations[0][0, :, :, output_vector_idx]

    for idx in range(1, len(activations)):
        # averaging of feature maps and element wise multiplication with previous layer's salient map
        salient_map = np.multiply(salient_map, np.average(activations[idx][0, :, :, :], axis=2))
        if idx < len(activations) - 1:
            salient_map = cv2.resize(salient_map, activations[idx + 1][0, :, :, 0].shape[::-1])
    salient_map = cv2.resize(salient_map, obs.shape[:2])

    # Saliency values are sometimes negative (if the output was negative)
    salient_map = np.abs(salient_map)
    # Scale to the 0.0-1.0 range
    if np.max(salient_map) != np.min(salient_map):
        salient_map = (salient_map - np.min(salient_map))/(np.max(salient_map) - np.min(salient_map))
    else:
        salient_map = np.zeros_like(salient_map)

    action_out = activations[0]
    return salient_map, action_out


def display_salient_map(salient_map, obs, window_title="Saliency", frames_in_stack_to_be_displayed=(0, 1, 2)):
    obs_bgr = obs[..., [2, 1, 0, 5, 4, 3, 8, 7, 6]]
    saliency_heatmap = cv2.applyColorMap((salient_map * 255).astype(np.uint8), cv2.COLORMAP_JET) / 255.
    # saliency_heatmap = np.repeat(saliency_map[...,None], 3, axis=2)
    # saliency_heatmap = saliency_map[...,None] * [[[0., 0., 1.]]]
    to_merge_rows = []
    for i in frames_in_stack_to_be_displayed:
        obs_i = obs_bgr[..., 3 * i:3 * (i + 1)]
        saliency_heatmap_overlayed = cv2.addWeighted(obs_i, 0.5, saliency_heatmap, 0.5, 0)
        to_merge_rows.append(np.concatenate([saliency_heatmap, obs_i, saliency_heatmap_overlayed], axis=1))
    merged = np.concatenate(to_merge_rows, axis=0)
    # saliency_heatmap_overlayed = cv2.addWeighted(obs_bgr[..., :3], 0.5, saliency_heatmap, 0.5, 0)
    # merged = np.concatenate([saliency_heatmap, obs_bgr[...,:3], saliency_heatmap_overlayed], axis=1)
    cv2.imshow(window_title, merged)
    cv2.waitKey(1)

def display_salient_map2(salient_map, obs, window_title="Saliency", frames_in_stack_to_be_displayed=(0, 1, 2),
                         use_color_map=True, overlay_only=True):
    """
    :param salient_map: Saleient object map, normed between 0 and 1
    :param obs: Observations, or stack of 3 observations. Channel order must be RGB(RGBRGB).
                Float representation is expected
    :param window_title:
    :param frames_in_stack_to_be_displayed: If frame stacking is used, select which RGB frames should be displayed
    :param use_color_map: Display salient obj map as a colored heatmap, or highlight salient objects with a single blue
                          color (opacity of the overlay is varied based on salient map values)
    :param overlay_only: Display overlay only or also display the heatmap and the observation separately
    """
    if obs.shape[2] == 9:
        obs_bgr = obs[..., [2, 1, 0, 5, 4, 3, 8, 7, 6]]
    else:
        obs_bgr = obs[..., [2, 1, 0]]
        frames_in_stack_to_be_displayed = [0]

    salient_map = cv2.resize(salient_map, obs_bgr.shape[1::-1])

    if use_color_map:
        saliency_heatmap = cv2.applyColorMap(((1-salient_map) * 255).astype(np.uint8), cv2.COLORMAP_JET) / 255.
    else:
        saliency_heatmap = np.ones((salient_map.shape[0], salient_map.shape[1], 3))
        saliency_heatmap = saliency_heatmap * np.array([[[1., 0, 0]]])  # "Heatmap" color is blue
    saliency_heatmap = saliency_heatmap.astype(np.float32)
    to_merge_rows = []
    for i in frames_in_stack_to_be_displayed:
        obs_i = obs_bgr[..., 3 * i:3 * (i + 1)].astype(np.float32)
        saliency_heatmap_overlayed = cv2.addWeighted(obs_i * (1 - np.power(salient_map[..., None], 2)), 1.0,
                                                     saliency_heatmap * np.power(salient_map[..., None], 2), 1.0, 0)
        if overlay_only:
            to_merge_rows.append(saliency_heatmap_overlayed)
        else:
            to_merge_rows.append(np.concatenate([saliency_heatmap, obs_i, saliency_heatmap_overlayed], axis=1))
    merged = np.concatenate(to_merge_rows, axis=0)
    # merged = cv2.resize(merged, tuple(np.array(merged.shape[:2]) * 4), interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow(window_title, merged)
    cv2.waitKey(1)
    return merged
