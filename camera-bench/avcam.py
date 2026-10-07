"""One AVFoundation camera, picked by name, behind the cv2.VideoCapture calls bench.py makes.

OpenCV's AVFoundation backend opens a camera by its position in a list that macOS reorders, even
between two opens in one process: on 2026-10-07, with the MacBook camera and a DJI Osmo Action 4
plugged in, OpenCV index 0 was the MacBook camera in one open and the Osmo in the next. So a camera
asked for by name is opened here, by the device itself, and frames come straight from AVFoundation.

    cam = AVCamera(find("osmo"), 1280, 720)
    ok, frame = cam.read()          # BGR uint8, the newest frame (late frames are dropped)
"""
import threading

import AVFoundation as AVF
import CoreMedia
import numpy as np
import objc
import Quartz
import libdispatch
from Foundation import NSObject

CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = 3, 4, 5


def devices():
    session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        [AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera, AVF.AVCaptureDeviceTypeExternal],
        AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified)
    return list(session.devices())


def names():
    return [str(d.localizedName()) for d in devices()]


def find(spec):
    """The camera for --camera: auto = a USB (UVC) camera if one is plugged in, else the Mac's default
    camera; anything else = the first camera whose name contains it (case ignored). None if no match."""
    devs = devices()
    if spec == "auto":
        for d in devs:
            if str(d.modelID()).startswith("UVC Camera"):
                return d
        return AVF.AVCaptureDevice.defaultDeviceWithMediaType_(AVF.AVMediaTypeVideo)
    for d in devs:
        if spec.lower() in str(d.localizedName()).lower():
            return d
    return None


class _Sink(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
    def initWithCamera_(self, cam):
        self = objc.super(_Sink, self).init()
        if self is not None:
            self.cam = cam
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        pb = CoreMedia.CMSampleBufferGetImageBuffer(sample)
        if pb is None:
            return
        Quartz.CVPixelBufferLockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            h, w = Quartz.CVPixelBufferGetHeight(pb), Quartz.CVPixelBufferGetWidth(pb)
            row = Quartz.CVPixelBufferGetBytesPerRow(pb)
            buf = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(row * h)
            frame = np.frombuffer(buf, np.uint8).reshape(h, row // 4, 4)[:, :w, :3].copy()  # BGRA -> BGR
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
        self.cam._put(frame)


class AVCamera:
    """Enough of cv2.VideoCapture for bench.py: read, get(3/4/5), set(5), isOpened, release."""

    def __init__(self, device, width, height):
        self.device, self.name = device, str(device.localizedName())
        self._cond = threading.Condition()
        self._frame, self._seq, self._read_seq = None, 0, 0
        self.session = AVF.AVCaptureSession.alloc().init()
        inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        self._ok = inp is not None and self.session.canAddInput_(inp)
        if not self._ok:
            return
        self.session.addInput_(inp)
        out = AVF.AVCaptureVideoDataOutput.alloc().init()
        out.setAlwaysDiscardsLateVideoFrames_(True)
        # Scaled to the asked size by AVFoundation, as OpenCV did: the MacBook camera sends 1080p whatever
        # format is asked for, and downstream sizes (CLAHE tiles, preview) were tuned on 1280x720.
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA,
                               Quartz.kCVPixelBufferWidthKey: width, Quartz.kCVPixelBufferHeightKey: height})
        self._sink = _Sink.alloc().initWithCamera_(self)
        self._queue = libdispatch.dispatch_queue_create(b"camera-bench.frames", None)
        out.setSampleBufferDelegate_queue_(self._sink, self._queue)
        self._ok = self.session.canAddOutput_(out)
        if self._ok:
            self.session.addOutput_(out)
            self.session.startRunning()

    def _put(self, frame):
        with self._cond:
            self._frame, self._seq = frame, self._seq + 1
            self._cond.notify_all()

    def read(self, timeout=1.0):
        """The newest frame not read yet; waits for one up to `timeout` s. (False, None) when none came."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._read_seq, timeout):
                return False, None
            self._read_seq = self._seq
            return True, self._frame

    def isOpened(self):
        return self._ok

    def get(self, prop):
        if prop == CAP_PROP_FPS:
            d = self.device.activeVideoMinFrameDuration()
            return d.timescale / d.value if d.value else 0.0
        f = self._frame
        if f is None:
            dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(self.device.activeFormat().formatDescription())
            return float(dims.width if prop == CAP_PROP_FRAME_WIDTH else dims.height)
        return float(f.shape[1] if prop == CAP_PROP_FRAME_WIDTH else f.shape[0])

    def set(self, prop, value):
        """Only the frame rate can be pinned, and only to a rate the active format allows: a camera with
        a fixed rate (the Osmo: 30.000 exactly) throws an Objective-C exception for anything else."""
        if prop != CAP_PROP_FPS or value <= 0:
            return False
        want = CoreMedia.CMTimeMake(1, int(round(value)))
        ranges = self.device.activeFormat().videoSupportedFrameRateRanges()
        if not any(CoreMedia.CMTimeCompare(r.minFrameDuration(), want) <= 0 <= CoreMedia.CMTimeCompare(r.maxFrameDuration(), want)
                   for r in ranges):
            return False
        if not self.device.lockForConfiguration_(None)[0]:
            return False
        self.device.setActiveVideoMaxFrameDuration_(want)  # max first: min may not exceed it
        self.device.setActiveVideoMinFrameDuration_(want)
        self.device.unlockForConfiguration()
        return True

    def release(self):
        if self._ok:
            self.session.stopRunning()
            self._ok = False
