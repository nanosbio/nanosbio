import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.libreria1 import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


@torch.no_grad()
def detect(weights, source, img_size, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf, save_crop,
           nosave, classes, agnostic_nms, augment, update, project, name, exist_ok, line_thickness, hide_labels,
           hide_conf):
    source, weights, view_img, save_txt, imgsz = source, weights, view_img, save_txt, img_size
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    salida = "Invalid Fingerprint"
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                   max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            # print(det)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # print(n)
                # Write results
                # print(end="\n")
                i=0
                count0 = 0
                count1 = 0
                for *xyxy, conf, cls in reversed(det):
                    i=i+1
                    """
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line)
                    """
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # print(label)
                        # print(names[c],conf)

                        if names[c] == "fake-a" or names[c] == "fake-b" or names[c] == "fake-d" or \
                            names[c] == "fake-f" or names[c] == "fake-g" or names[c] == "fake-h" or \
                            names[c] == "fake-i" or names[c] == "fake-j" or names[c] == "fake-k" or names[c] == "fake-l" or \
                            names[c] == "fake-m" or names[c] == "fake-n" or names[c] == "fake-o" or \
                            names[c] == "fake-p" or names[c] == "fake-q" or names[c] == "fake-r" or names[c] == "fake-s" or \
                            names[c] == "fake-t" or (names[c] == "fake-u" and conf > 0.82):
                            count0 = count0 + 1

                        if names[c] == "fake-a" or names[c] == "fake-b" or names[c] == "fake-c" or names[
                            c] == "fake-d" or \
                                names[c] == "fake-f" or names[c] == "fake-g" or names[c] == "fake-h" or names[
                            c] == "fake-i" or \
                                names[c] == "fake-j" or names[c] == "fake-k" or names[c] == "fake-l" or names[
                            c] == "fake-m" or names[c] == "fake-n" or names[c] == "fake-o" or names[c] == "fake-p" or\
                            names[c] == "fake-q" or names[c] == "fake-r" or names[c] == "fake-s" or names[c] == "fake-t" or\
                            names[c] == "fake-u":
                            count1 = count1 + 1
                        """"
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        """
                if count0 > 0 or count1 == i:
                    salida = "Invalid Fingerprint"
                else:
                    salida = "Accepted Fingerprint"
                #print(salida)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            # print(f'Done. {s}')
            """
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            """
    '''if save_txt or save_img:
        s = f"{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")'''

    return salida


def evaluarimagen(image):
    weights = "best.pt"
    source = image
    img_size = 640
    conf_thres = 0.42
    iou_thres = 0.45
    max_det = 4
    device = ''
    view_img = False
    save_txt = True
    save_conf = True
    save_crop = True
    nosave = True
    classes = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ,17 ,18, 19, 20
    agnostic_nms = True
    augment = True
    update = True
    project = 'runs/detect'
    name = 'exp'
    exist_ok = True
    line_thickness = 3
    hide_labels = False
    hide_conf = False
    #check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
    try:
        time.sleep(0)
        salida = detect(weights, source, img_size, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf,
                  save_crop, nosave, classes, agnostic_nms, augment, update, project, name, exist_ok, line_thickness,
                  hide_labels, hide_conf)

        return salida
    
    except:
        return "Processing error"
