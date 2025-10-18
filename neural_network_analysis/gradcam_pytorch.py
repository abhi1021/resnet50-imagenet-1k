"""
Wrapper for pytorch-grad-cam (grad-cam) to visualize Grad-CAM on CIFAR images.

Install dependency:
    pip install grad-cam

Usage:
    from neural_network_analysis.gradcam_pytorch import show_gradcam_batch
    show_gradcam_batch(model, device, dataloader, classes)

This will pick the first batch from the dataloader and display images, heatmaps and overlays.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception as e:
    raise ImportError("Please install pytorch-grad-cam: pip install grad-cam") from e


def _find_target_layer(model):
    # heuristic: last Conv2d
    import torch.nn as nn
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    return None


def show_gradcam_batch(model, device, loader, classes=None, target_layer=None, n=4):
    model.to(device)
    model.eval()

    if target_layer is None:
        target_layer = _find_target_layer(model)
        if target_layer is None:
            raise RuntimeError("Couldn't find a Conv2d target layer automatically. Pass target_layer explicitly.")

    data_iter = iter(loader)
    images, labels = next(data_iter)
    images = images[:n].to(device)
    labels = labels[:n].to(device)

    # prepare images for visualization (unnormalize)
    mean = np.array([0.5071, 0.4867, 0.4408]).reshape(3,1,1)
    std = np.array([0.2675, 0.2565, 0.2761]).reshape(3,1,1)
    imgs_np = images.detach().cpu().numpy()
    imgs_vis = []
    for i in range(imgs_np.shape[0]):
        im = imgs_np[i]
        im = (im * std) + mean
        im = np.transpose(im, (1,2,0))
        imgs_vis.append(im)

    # Try to detect installed grad-cam version (helpful for debugging)
    try:
        from importlib.metadata import version as _get_version
    except Exception:
        try:
            from importlib_metadata import version as _get_version
        except Exception:
            _get_version = None

    gradcam_version = None
    if _get_version is not None:
        try:
            gradcam_version = _get_version('grad-cam')
        except Exception:
            gradcam_version = None

    if gradcam_version is not None:
        print(f"[grad-cam] detected version: {gradcam_version}")

    # pytorch-grad-cam expects a list of target layers (target_layers)
    # use_cuda=True works for both CUDA and MPS devices
    use_gpu = device.type in ['cuda', 'mps']
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_gpu)
    targets = [ClassifierOutputTarget(int(labels[i].item())) for i in range(len(labels))]

    # Try multiple call signatures to handle different grad-cam versions
    call_errors = []
    grayscale_cams = None
    call_attempts = [
        lambda: cam(input_tensor=images, targets=targets),
        lambda: cam(input_tensor=images),
        lambda: cam(images, targets=targets),
        lambda: cam(images),
        lambda: cam(input_tensor=images, targets=None),
    ]

    for attempt in call_attempts:
        try:
            grayscale_cams = attempt()
            break
        except Exception as e:
            call_errors.append(str(e))

    if grayscale_cams is None:
        msg = 'Failed to invoke GradCAM with tried signatures. Errors:\n' + '\n'.join(call_errors)
        if gradcam_version is None:
            msg += '\nAlso failed to detect installed grad-cam version. Try: pip install grad-cam>=1.4.3'
        else:
            msg += f'\nDetected grad-cam version: {gradcam_version}. If incompatible, try: pip install -U grad-cam'
        raise RuntimeError(msg)

    # Normalize the returned value to a numpy array of shape (B, H, W)
    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, (list, tuple)):
            # If it's a list of arrays/tensors, try to convert/stack sensibly
            arrs = []
            for elem in x:
                if isinstance(elem, (np.ndarray, torch.Tensor)):
                    arrs.append(_to_numpy(elem))
                elif isinstance(elem, (list, tuple)) and len(elem) > 0:
                    arrs.append(_to_numpy(elem[0]))
                else:
                    raise RuntimeError(f'Unexpected element type in GradCAM return: {type(elem)}')
            # If arrs contains a single 3D array (B,H,W), return it
            if len(arrs) == 1 and arrs[0].ndim == 3:
                return arrs[0]
            # If arrs is a list of 2D arrays (H,W) per sample, stack into (B,H,W)
            if all(a.ndim == 2 for a in arrs):
                return np.stack(arrs, axis=0)
            # If arrs holds multiple layers (n_layers, B, H, W) or (B, n_layers, H, W), try to merge
            try:
                return np.stack(arrs, axis=0)
            except Exception:
                raise RuntimeError('Unable to normalize GradCAM return to a numpy array')
        raise RuntimeError(f'Unsupported GradCAM return type: {type(x)}')

    grayscale_cams = _to_numpy(grayscale_cams)

    # If we got 4 dims (possible shapes: B, n_layers, H, W) or (n_layers, B, H, W), convert to (B,H,W)
    if grayscale_cams.ndim == 4:
        b = imgs_np.shape[0]
        if grayscale_cams.shape[0] == b:
            # (B, n_layers, H, W) -> average over layers
            grayscale_cams = grayscale_cams.mean(axis=1)
        elif grayscale_cams.shape[1] == b:
            # (n_layers, B, H, W) -> take mean over layers
            grayscale_cams = grayscale_cams.mean(axis=0)
        else:
            # fallback: mean across first axis
            grayscale_cams = grayscale_cams.mean(axis=0)

    if grayscale_cams.ndim != 3:
        raise RuntimeError(f'After normalization, expected grayscale_cams to be 3D (B,H,W); got shape {grayscale_cams.shape}')

    plt.figure(figsize=(12, 3*n))
    for i in range(n):
        gcam = grayscale_cams[i]
        img_vis = imgs_vis[i]
        cam_image = show_cam_on_image(img_vis, gcam, use_rgb=True)

        ax = plt.subplot(n, 3, 3*i+1)
        plt.imshow(img_vis)
        plt.title(f"GT: {classes[labels[i].item()] if classes is not None else labels[i].item()}")
        plt.axis('off')

        ax = plt.subplot(n, 3, 3*i+2)
        plt.imshow(gcam, cmap='jet')
        plt.title('Grad-CAM')
        plt.axis('off')

        ax = plt.subplot(n, 3, 3*i+3)
        plt.imshow(cam_image)
        # compute predicted class
        with torch.no_grad():
            out = model(images[i:i+1])
            pred = int(out.argmax(dim=1).item())
        plt.title(f"Pred: {classes[pred] if classes is not None else pred}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
