"""Bald head maker."""

import cv2
import sys

import torch
import numpy as np

from torchvision.transforms.functional import normalize

from bald_head_maker.facelib.utils.face_restoration_helper import FaceRestoreHelper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make(self, facerestore_model, image, facedetection, codeformer_fidelity):
    print(f"\tStarting restore_face with codeformer_fidelity: {codeformer_fidelity}")
    facerestore_model.to(device)
    if self.face_helper is None:
        self.face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=facedetection,
            save_ext="png",
            use_parse=True,
            device=device,
        )

    image_np = 255.0 * image.cpu().numpy()

    total_images = image_np.shape[0]
    out_images = np.ndarray(shape=image_np.shape)

    for i in range(total_images):
        cur_image_np = image_np[i, :, :, ::-1]

        original_resolution = cur_image_np.shape[0:2]

        if facerestore_model is None or self.face_helper is None:
            return image

        self.face_helper.clean_all()
        self.face_helper.read_image(cur_image_np)
        self.face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        self.face_helper.align_warp_face()

        restored_face = None
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = facerestore_model(cropped_face_t, w=codeformer_fidelity)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f"\tFailed inference for CodeFormer: {error}", file=sys.stderr)
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        self.face_helper.get_inverse_affine(None)

        restored_img = self.face_helper.paste_faces_to_input_image()
        restored_img = restored_img[:, :, ::-1]

        if original_resolution != restored_img.shape[0:2]:
            restored_img = cv2.resize(
                restored_img,
                (0, 0),
                fx=original_resolution[1] / restored_img.shape[1],
                fy=original_resolution[0] / restored_img.shape[0],
                interpolation=cv2.INTER_LINEAR,
            )

        self.face_helper.clean_all()

        # restored_img = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)

        out_images[i] = restored_img

    restored_img_np = np.array(out_images).astype(np.float32) / 255.0
    restored_img_tensor = torch.from_numpy(restored_img_np)
    return (restored_img_tensor,)
