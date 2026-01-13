import cv2
import librosa
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from numpy.typing import NDArray
from python_speech_features import logfbank
from transformers import FeatureExtractionMixin
from transformers.feature_extraction_utils import BatchFeature

mp_face_mesh = mp.solutions.face_mesh


class AVHubertFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["input_values", "pixel_values"]

    def __init__(
        self,
        max_sample_size: int | None = None,
        normalize: bool = True,
        stack_order_audio: int = 4,
        image_crop_size: int = 88,
        image_mean: float = 0.421,
        image_std: float = 0.165,
        sr: int = 16_000,
        static_image_mode: bool = False,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        landmark_indices: tuple[int, ...] = (5, 411, 199, 187),  # (top, right, bottom, left) of mouth
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_sample_size = max_sample_size
        self.normalize = normalize
        self.stack_order_audio = stack_order_audio
        self.image_crop_size = image_crop_size
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.CenterCrop(image_crop_size),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([image_mean], [image_std]),
            ]
        )
        self.sr = sr
        self.static_image_mode = static_image_mode
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmark_indices = landmark_indices

    def _load_video(self, video: str | NDArray[np.uint8], extract_mouth: bool = False) -> torch.FloatTensor:
        """Input video must be in RGB format if type is numpy array."""
        if isinstance(video, str):
            cap = cv2.VideoCapture(video)
            frames = []
            for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if not ret:
                    break
                if not extract_mouth:  # Already extracted mouth
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                else:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_np = np.stack(frames, axis=0)
        else:
            frames_np = video
            if not extract_mouth:  # Already extracted mouth
                frames_np = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_np], axis=0)

        if extract_mouth:
            frames_np = self._extract_mouth(frames_np)

        return torch.from_numpy(frames_np).unsqueeze(dim=1)

    def _extract_mouth(self, frames: NDArray[np.uint8]) -> NDArray[np.uint8]:
        mouth_frames = []
        top_idx, right_idx, bottom_idx, left_idx = self.landmark_indices
        with mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as face_mesh:
            for frame in frames:
                res = face_mesh.process(frame)
                if res.multi_face_landmarks is None or len(res.multi_face_landmarks) == 0:
                    mouth_frames.append(np.zeros([self.image_crop_size, self.image_crop_size], dtype=np.uint8))
                    continue
                landmarks = res.multi_face_landmarks[0].landmark
                top = landmarks[top_idx]
                left = landmarks[left_idx]
                right = landmarks[right_idx]
                bottom = landmarks[bottom_idx]

                H, W = frame.shape[:2]
                xmax = max(top.x, left.x, right.x, bottom.x)
                ymax = max(top.y, left.y, right.y, bottom.y)
                xmin = min(top.x, left.x, right.x, bottom.x)
                ymin = min(top.y, left.y, right.y, bottom.y)

                patch_size = max((xmax - xmin) * W, (ymax - ymin) * H)  # To extract square region
                half = int(patch_size / 2)
                y_center = int(ymin * H) + int(((ymax - ymin) / 2) * H)
                x_center = int(xmin * W) + int(((xmax - xmin) / 2) * W)
                lip = frame[
                    y_center - half : y_center + half,
                    x_center - half : x_center + half,
                    :,
                ]
                try:
                    lip = cv2.resize(lip, (self.image_crop_size, self.image_crop_size))
                except Exception:
                    lip = np.zeros([self.image_crop_size, self.image_crop_size, 3], dtype=np.uint8)
                mouth_frames.append(cv2.cvtColor(lip, cv2.COLOR_RGB2GRAY))
        return np.stack(mouth_frames, axis=0)

    def _load_audio(self, audio: str | NDArray[np.float32]) -> torch.FloatTensor:
        def stacker(feats, stack_order):
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
            return feats

        sr = None
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=16_000)
        if sr is None:
            sr = self.sr
        fbank = logfbank(audio, samplerate=sr).astype(np.float32)
        fbank = stacker(fbank, self.stack_order_audio)
        return torch.from_numpy(fbank)

    def _align_time_steps(
        self, audio: list[torch.FloatTensor], video: list[torch.FloatTensor]
    ) -> tuple[list[torch.FloatTensor], list[torch.FloatTensor]]:
        aligned_indices = []
        for sample_audio, sample_video in zip(audio, video):
            diff = len(sample_audio) - len(sample_video)
            if diff != 0:
                aligned_indices.append(
                    torch.arange(0, len(sample_audio)).float() * len(sample_video) / len(sample_audio)
                )
            else:
                aligned_indices.append(torch.arange(0, len(sample_audio)))
        return (
            audio,
            [
                sample[torch.clamp(torch.floor(indices), max=sample.shape[0] - 1).long()]
                for sample, indices in zip(video, aligned_indices)
            ],
        )

    def __call__(
        self,
        raw_audio: NDArray[np.float32] | str | list[NDArray[np.float32]] | list[str] | None = None,
        raw_video: NDArray[np.uint8] | str | list[NDArray[np.uint8]] | list[str] | None = None,
        extract_mouth: bool = False,
        **kwargs,
    ) -> BatchFeature:
        if not isinstance(raw_audio, list):
            raw_audio = [raw_audio]
        if not isinstance(raw_video, list):
            raw_video = [raw_video]

        audio = [self._load_audio(sample) if sample is not None else None for sample in raw_audio]
        video = [self._load_video(sample, extract_mouth) if sample is not None else None for sample in raw_video]
        for batch_idx in range(len(audio)):
            sample_a = audio[batch_idx]
            sample_v = video[batch_idx]
            assert sample_a is not None or sample_v is not None
            if sample_a is None:
                sample_a = torch.zeros((sample_v.shape[0], 26 * self.stack_order_audio))
                audio[batch_idx] = sample_a
            elif sample_v is None:  # 25 fps
                sample_v = torch.zeros((sample_a.shape[0], 1, self.image_crop_size, self.image_crop_size))
                video[batch_idx] = sample_v

        audio, video = self._align_time_steps(audio, video)
        max_length = max(len(data) for data in audio)
        input_values = []
        pixel_values = []
        attention_mask = []
        for feat_audio, feat_video in zip(audio, video):
            remainder_length = max_length - len(feat_audio)
            audio_remainder = torch.zeros(
                size=(remainder_length,) + feat_audio.size()[1:],
                dtype=feat_audio.dtype,
            )
            video_remainder = torch.zeros(
                size=(remainder_length,) + feat_video.size()[1:],
                dtype=feat_video.dtype,
            )

            feat_audio = torch.cat((feat_audio, audio_remainder))
            feat_video = torch.cat((feat_video, video_remainder))
            if self.max_sample_size:
                feat_audio = feat_audio[: self.max_sample_size]
                feat_video = feat_video[: self.max_sample_size]
            attn_mask = torch.ones(max_length)
            attn_mask[max_length - remainder_length :] = 0

            input_values.append(feat_audio)
            pixel_values.append(feat_video)
            attention_mask.append(attn_mask)

        input_values = torch.stack(input_values)
        batch = BatchFeature(
            {
                "input_values": (
                    F.layer_norm(input_values, input_values.shape[2:]) if self.normalize else input_values
                ),
                "pixel_values": self.transforms(torch.stack(pixel_values)),
                "attention_mask": torch.stack(attention_mask),
            }
        )
        return batch

    def to_dict(self):
        output = super().to_dict()
        output["transforms"] = self._transforms_to_dict(output["transforms"])
        return output

    def _transforms_to_dict(self, transforms: transforms.Compose):
        output = []
        for component in transforms.__dict__["transforms"]:
            name = component.__class__.__name__
            component_dict = {"transforms_type": name}
            for k, v in component.__dict__.items():
                if k.startswith("_"):
                    continue
                component_dict[k] = str(v)
            output.append(component_dict)
        return output
