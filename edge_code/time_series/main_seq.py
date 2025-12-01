#!/usr/bin/env python3
# Copyright 2025 NXP
# SPDX-License-Identifier: Apache-2.0
"""i.MX Sequential Gesture Recognition"""
import os
import sys
import time
import logging
import argparse

import cv2

import hand_tracker_seq
import gesture_classifier_seq
from app_utils import drawkit_seq

# GoPoint
if os.path.isdir("/opt/gopoint-apps/scripts/machine_learning/imx_gesture_recognition"):
    sys.path.append("/opt/gopoint-apps/scripts/machine_learning/imx_gesture_recognition")

SEQUENCE_LENGTH = 10
MOVEMENT_THRESHOLD = 1.0
RESET_COOLDOWN_DURATION = 4.0

GESTURE_NAMES = {
    0: "Swipe Up",
    1: "Swipe Down",
    2: "Swipe Left",
    3: "Swipe Right",
    4: "Zoom In",
    5: "Zoom Out",
    6: "Rotate Clockwise",
    7: "Rotate Counter Clockwise"
}


def run(stream, args):
    """Run Sequential Hand Gesture Recognition task"""
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video stream or file: {stream}")

    tracker = hand_tracker_seq.HandTracker(
        palm_detection_model=args.palm_model,
        hand_landmark_model=args.hand_landmark_model,
        anchors=args.anchors,
        external_delegate=args.external_delegate_path,
        num_hands=args.num_hands,
    )

    classifier = gesture_classifier_seq.Classifier(model=args.classification_model)
    
    # Initialize sequence buffer
    sequence_buffer = gesture_classifier_seq.SequenceBuffer(
        sequence_length=SEQUENCE_LENGTH,
        feature_size=63,
        movement_threshold=MOVEMENT_THRESHOLD
    )
    
    cv2.namedWindow("i.MX Sequential Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    
    in_cooldown = False
    cooldown_frames = 0
    cooldown_duration = 5
    last_prediction_result = None
    
    # Warm-up: wait for stable hand detection before collecting
    warmup_frames = 0
    warmup_required = 3  # Require 3 consecutive detections before starting to collect
    
    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        current_prediction = None
        has_movement = True
        
        # Calculate FPS
        fps_frame_count += 1
        fps_elapsed = time.time() - fps_start_time
        if fps_elapsed >= 1.0:  # Update FPS every second
            current_fps = fps_frame_count / fps_elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
        
        if in_cooldown:
            cooldown_frames -= 1
            if cooldown_frames <= 0:
                sequence_buffer.reset("Prediction completed")
                in_cooldown = False
        
        status = sequence_buffer.get_status()
        
        if not in_cooldown and not status['in_reset_cooldown']:
            detections = tracker(frame)
            
            if detections:
                for results in detections:
                    landmarks, hand_bbox, handedness = results
                    drawkit_seq.draw_landmarks(landmarks, frame)
                    drawkit_seq.draw_handbbox(hand_bbox, frame)
                    
                    # Warm-up: only start collecting after stable detections
                    if warmup_frames < warmup_required:
                        warmup_frames += 1
                    else:
                        sequence_buffer.add_frame(landmarks)
            else:
                # Reset warm-up counter if hand is lost
                warmup_frames = 0
                sequence_buffer.add_null_frame()
            
            status = sequence_buffer.get_status()
            
            if status['is_ready'] and not in_cooldown and not status['in_reset_cooldown']:
                has_movement = status['has_significant_movement']
                sequence = sequence_buffer.get_sequence()
                
                if sequence is not None:
                    current_prediction = classifier(sequence)
                    
                    last_prediction_result = {
                        'predicted_class': current_prediction['predicted_class'],
                        'max_confidence': current_prediction['max_confidence'],
                        'entropy': current_prediction['entropy'],
                        'is_unknown': current_prediction['is_unknown'],
                        'has_movement': has_movement
                    }
                    
                    # Reset buffer immediately before starting cooldown
                    sequence_buffer.reset("Prediction completed")
                    warmup_frames = 0  # Reset warm-up counter
                    in_cooldown = True
                    cooldown_frames = cooldown_duration
        
        drawkit_seq.draw_overlay(frame, status, current_prediction, has_movement, 
                                 last_prediction_result, GESTURE_NAMES, 
                                 SEQUENCE_LENGTH, RESET_COOLDOWN_DURATION)
        
        # Draw FPS on frame
        drawkit_seq.draw_fps(current_fps, frame)

        cv2.imshow("i.MX Sequential Gesture Recognition", frame)
        
        # Explicit 10ms delay between frames
        # time.sleep(0.01)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="i.MX Sequential Gesture Recognition showcases the Machine "
        "Learning (ML) capabilities of the i.MX SoCs (i.MX 93 and i.MX 8M Plus) "
        "using the available Neural Processing Unit (NPU) to accelerate two "
        "Deep Learning vision-based models. Together, these models detect up to "
        "two hands present in the scene and predict 21 3D-keypoints that are used "
        "to recognize dynamic hand gestures using sequences of landmarks with a "
        "1D Convolutional Neural Network."
    )

    parser.add_argument(
        "-f",
        "--file",
        metavar="file",
        type=str,
        help="Input file. It can be an image or a video.",
    )
    parser.add_argument(
        "-d",
        "--device",
        metavar="device",
        type=str,
        help="Camera device. Please provide the camera device " "as /dev/videoN.",
    )
    parser.add_argument(
        "-e",
        "--external_delegate_path",
        metavar="external delegate",
        type=str,
        help="Path to external delegate for HW acceleration.",
    )
    parser.add_argument(
        "--logging_level",
        metavar="logging level",
        type=int,
        default=logging.WARNING,
        help="Logging level priority.",
    )

    parser.add_argument(
        "--palm_model",
        metavar="palm model",
        type=str,
        required=True,
        help="Path to palm detection model.",
    )
    parser.add_argument(
        "--hand_landmark_model",
        metavar="hand landmark model",
        type=str,
        required=True,
        help="Path to hand landmark model.",
    )
    parser.add_argument(
        "--classification_model",
        metavar="classification model",
        type=str,
        required=True,
        help="Path to sequential classification model.",
    )
    parser.add_argument(
        "--anchors",
        metavar="anchors",
        type=str,
        required=True,
        help="Path to anchors file.",
    )

    parser.add_argument(
        "--num_hands",
        metavar="Number of hands",
        type=int,
        default=1,
        help="Max number of hands that will be detected [1, 2]",
    )

    args = parser.parse_args()
    source = args.file
    if source:
        if not os.path.isfile(args.file):
            raise FileNotFoundError(
                "Source file does not exists. Please provide"
                " a valid source. You can check"
                " python3 main_seq.py --help for more details."
            )

    elif args.device:
        source = args.device

    if args.external_delegate_path:
        if not os.path.isfile(args.external_delegate_path):
            raise FileNotFoundError(f"File {args.external_delegate_path} not found.")

    if not args.num_hands in [1, 2]:
        raise ValueError("The Number of hands must be 1 or 2.")

    if not os.path.isfile(args.palm_model):
        raise FileNotFoundError(f"File {args.palm_model} not found.")
    if not os.path.isfile(args.hand_landmark_model):
        raise FileNotFoundError(f"File {args.hand_landmark_model} not found.")
    if not os.path.isfile(args.classification_model):
        raise FileNotFoundError(f"File {args.classification_model} not found.")
    if not os.path.isfile(args.anchors):
        raise FileNotFoundError(f"File {args.anchors} not found.")

    logging.basicConfig(level=args.logging_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger()

    logging.info("Source: %s", source)
    logging.info("Palm detection: %s", args.palm_model)
    logging.info("Hand landmark: %s", args.hand_landmark_model)
    logging.info("Classification model: %s", args.classification_model)
    logging.info("External delegate: %s", args.external_delegate_path)
    logging.info("Number of hands: %d", args.num_hands)

    run(source, args)